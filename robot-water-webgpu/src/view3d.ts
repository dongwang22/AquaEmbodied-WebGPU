import type { FluidGPU } from './gpu';
import volumeCode from './shaders/volume.wgsl?raw';
import robotViewCode from './shaders/robot-view.wgsl?raw';
import captureCode from './shaders/capture-frame.wgsl?raw';
import { interpolateParticles, type PlaybackFrame } from './playback-frame';
export type { PlaybackFrame } from './playback-frame';

export interface RobotViewOptions {
  waterOpacity?:number;
  targetFps?:number;
  pixelBudget?:number;
  cameraYaw?:number;
  cameraPitch?:number;
  cameraDistance?:number;
  cameraCenter?:[number,number,number];
}

type V3=[number,number,number];
const unit=(v:V3):V3=>{const l=Math.hypot(...v);return v.map(x=>x/l) as V3;};
const cross=(a:V3,b:V3):V3=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];

/** Direct GPU volume rendering, with orbit controls independent of simulation. */
export class View3D {
  private readonly context:GPUCanvasContext;
  private readonly parameters:GPUBuffer;
  private readonly particles:GPUBuffer;
  private readonly particleGroups:GPUBuffer;
  private readonly volume:GPUTexture;
  private compute!:GPUComputePipeline;
  private renderPipeline!:GPURenderPipeline;
  private copyGroup!:GPUBindGroup;
  private renderGroup!:GPUBindGroup;
  private liveRenderGroup!:GPUBindGroup;
  private sampler!:GPUSampler;
  private capturePipeline!:GPUComputePipeline;
  private captureResources?:{packed:GPUBuffer;staging:GPUBuffer;group:GPUBindGroup;bytes:number};
  private capturing=false;
  private playbackSlots?:{texture:GPUTexture;view:GPUTextureView;frame?:PlaybackFrame}[];
  private playbackPair='';
  private replaying=false;
  private replayAlpha=0;
  private readonly abort=new AbortController();
  private readonly resize:ResizeObserver;
  private data=new Float32Array(24);
  private yaw=2.35;
  private pitch=.57;
  private distance=670;
  private center:V3=[256,80,26];
  private dirty=true;
  private volumeDirty=true;
  private disposed=false;
  private inFlight=false;
  private frame=0;
  private lastFrame=0;
  private pointer:{id:number,x:number,y:number,pan:boolean}|undefined;
  private field=0;
  private opacity=.8;
  frames=0;
  get camera(){return{yaw:this.yaw,pitch:this.pitch,distance:this.distance,center:[...this.center]};}

  private constructor(private readonly canvas:HTMLCanvasElement,private readonly fluid:FluidGPU,private readonly onError:(e:unknown)=>void,private readonly options?:RobotViewOptions){
    const device=fluid.device;
    this.context=canvas.getContext('webgpu')!;
    if(!this.context)throw new Error('Unable to create the 3D WebGPU canvas.');
    this.context.configure({device,format:navigator.gpu.getPreferredCanvasFormat(),alphaMode:'opaque'});
    this.parameters=device.createBuffer({size:96,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.particles=device.createBuffer({size:fluid.particleCount*64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    this.particleGroups=device.createBuffer({size:Math.ceil(fluid.particleCount/20)*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    this.volume=device.createTexture({label:'Visualization only: phi / speed / rho',size:[fluid.config.nx,fluid.config.ny,fluid.config.nz],dimension:'3d',format:'rgba16float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});
    this.resize=new ResizeObserver(()=>{this.dirty=true;});this.resize.observe(canvas);
    this.resetCamera();this.opacity=options?.waterOpacity??.8;this.controls();
  }
  static async create(canvas:HTMLCanvasElement,fluid:FluidGPU,onError:(e:unknown)=>void,options?:RobotViewOptions){
    const result=new View3D(canvas,fluid,onError,options);
    try{await result.compile();result.frame=requestAnimationFrame(result.tick);return result;}
    catch(e){result.destroy();throw e;}
  }
  private async compile(){
    const d=this.fluid.device;
    const modules=[d.createShaderModule({code:volumeCode}),d.createShaderModule({code:robotViewCode}),d.createShaderModule({code:captureCode})];
    for(const m of modules){const info=await m.getCompilationInfo();const errors=info.messages.filter(e=>e.type==='error');if(errors.length)throw new Error(errors.map(e=>`3D WGSL ${e.lineNum}: ${e.message}`).join('\n'));}
    this.compute=await d.createComputePipelineAsync({layout:'auto',compute:{module:modules[0],entryPoint:'copyVolume'}});
    this.capturePipeline=await d.createComputePipelineAsync({layout:'auto',compute:{module:modules[2],entryPoint:'captureFrame'}});
    this.renderPipeline=await d.createRenderPipelineAsync({layout:'auto',vertex:{module:modules[1],entryPoint:'vertex'},fragment:{module:modules[1],entryPoint:'fragment',targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},primitive:{topology:'triangle-list'}});
    const texture=this.volume.createView({dimension:'3d'});
    this.copyGroup=d.createBindGroup({layout:this.compute.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.parameters}},{binding:1,resource:{buffer:this.fluid.cellBuffer}},{binding:2,resource:texture}]});
    this.sampler=d.createSampler({minFilter:'linear',magFilter:'linear'});
    this.liveRenderGroup=this.makeRenderGroup(texture,texture);this.renderGroup=this.liveRenderGroup;
  }
  private makeRenderGroup(volume:GPUTextureView,nextVolume:GPUTextureView){
    return this.fluid.device.createBindGroup({layout:this.renderPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.parameters}},{binding:1,resource:{buffer:this.particles}},{binding:2,resource:volume},{binding:3,resource:this.sampler},{binding:4,resource:{buffer:this.particleGroups}},{binding:5,resource:nextVolume}]});
  }
  private uploadParticles(particles:Float32Array,orientations:Float32Array){
    const count=particles.length/12,p=new Float32Array(count*8),s=this.fluid.config.scale;
    for(let i=0;i<count;i++){
      for(let j=0;j<4;j++)p[i*8+j]=particles[i*12+j]/s;
      p.set(orientations.subarray(i*4,i*4+4),i*8+4);
    }
    this.fluid.device.queue.writeBuffer(this.particles,0,p);
    // Refit small particle AABBs on the CPU; skip whole groups for most rays.
    // Membership affects drawing cost only, never numerical interactions.
    const groups=new Float32Array(Math.ceil(count/20)*8);
    for(let first=0;first<count;first+=20){
      const g=Math.floor(first/20)*8,end=Math.min(first+20,count);
      const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
      for(let i=first;i<end;i++)for(let axis=0;axis<3;axis++){
        low[axis]=Math.min(low[axis],p[i*8+axis]-p[i*8+3]-.01);
        high[axis]=Math.max(high[axis],p[i*8+axis]+p[i*8+3]+.01);
      }
      groups.set([...low,first,...high,end],g);
    }
    this.fluid.device.queue.writeBuffer(this.particleGroups,0,groups);
  }
  update(particles:Float32Array,orientations:Float32Array){
    this.uploadParticles(particles,orientations);
    this.renderGroup=this.liveRenderGroup;this.replaying=false;this.replayAlpha=0;this.playbackPair='';
    this.volumeDirty=true;this.dirty=true;
  }
  updateRobot(bodies:Float32Array){
    if(bodies.length!==this.fluid.particleCount*16)throw new Error('Robot view/body layout mismatch.');
    const p=bodies.slice(),count=this.fluid.particleCount,s=this.fluid.config.scale;
    for(let i=0;i<count;i++){for(let j=0;j<4;j++)p[i*16+j]/=s;p[i*16+7]/=s;}
    const groups=new Float32Array(Math.ceil(count/20)*8);
    for(let first=0;first<count;first+=20){
      const end=Math.min(first+20,count),low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
      for(let i=first;i<end;i++)for(let d=0;d<3;d++){
        const extent=p[i*16+3]+Math.abs(p[i*16+12+d])*p[i*16+7];
        low[d]=Math.min(low[d],p[i*16+d]-extent);high[d]=Math.max(high[d],p[i*16+d]+extent);
      }
      groups.set([...low,first,...high,end],Math.floor(first/20)*8);
    }
    this.fluid.device.queue.writeBuffer(this.particles,0,p);this.fluid.device.queue.writeBuffer(this.particleGroups,0,groups);
    // Snapshot water beside this completed pose, before another simulation batch
    // can change cellBuffer. The independent render loop reads this matched pair.
    const cfg=this.fluid.config,d=this.fluid.device;
    this.data.set([cfg.nx,cfg.ny,cfg.nz,cfg.scale],16);
    d.queue.writeBuffer(this.parameters,0,this.data);
    const encoder=d.createCommandEncoder({label:'Completed robot / water snapshot'});
    const pass=encoder.beginComputePass();pass.setPipeline(this.compute);pass.setBindGroup(0,this.copyGroup);
    pass.dispatchWorkgroups(Math.ceil(cfg.nx/4),Math.ceil(cfg.ny/4),Math.ceil(cfg.nz/4));pass.end();
    d.queue.submit([encoder.finish()]);
    this.volumeDirty=false;this.dirty=true;
  }
  /** Record one compact visual frame after a completed simulation step. */
  async captureFrame(time:number,particles:Float32Array,orientations:Float32Array):Promise<PlaybackFrame>{
    if(this.disposed)throw new Error('The 3D view has been disposed.');
    if(this.capturing)throw new Error('Visual frame captures must be awaited in sequence.');
    if(!Number.isFinite(time)||time<0)throw new Error('Visual frame time must be finite and nonnegative.');
    if(particles.length!==this.fluid.particleCount*12||orientations.length!==this.fluid.particleCount*4)
      throw new Error('Visual frame particle count does not match the simulation.');
    const bodies=particles.slice(),rotations=orientations.slice();
    const d=this.fluid.device,count=this.fluid.cellCount;
    if(!this.captureResources){
      const bytes=Math.ceil(count/4)*4;
      const packed=d.createBuffer({label:'Compact visual frame',size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
      const staging=d.createBuffer({label:'Visual frame readback',size:bytes,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
      const group=d.createBindGroup({layout:this.capturePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.fluid.cellBuffer}},{binding:1,resource:{buffer:packed}}]});
      this.captureResources={packed,staging,group,bytes};
    }
    this.capturing=true;
    const {packed,staging,group,bytes}=this.captureResources;
    try{
      const encoder=d.createCommandEncoder({label:'Record visual sample'}),pass=encoder.beginComputePass();
      pass.setPipeline(this.capturePipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(bytes/4/128));pass.end();
      encoder.copyBufferToBuffer(packed,0,staging,0,bytes);d.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const phi=new Uint8Array(staging.getMappedRange().slice(0,count));
      return{time,phi,particles:bodies,orientations:rotations};
    }finally{if(staging.mapState==='mapped')staging.unmap();this.capturing=false;}
  }
  /** Show an interpolated pair at wall-clock time; never advances the solver. */
  showPlayback(a:PlaybackFrame,b:PlaybackFrame,alpha:number){
    if(this.disposed)return;
    if(a.phi.length!==this.fluid.cellCount||b.phi.length!==this.fluid.cellCount)
      throw new Error('Visual frame grid does not match the simulation.');
    if(!Number.isFinite(alpha))throw new Error('Playback interpolation must be finite.');
    const {particles,orientations}=interpolateParticles(a,b,alpha);this.uploadParticles(particles,orientations);
    const d=this.fluid.device,cfg=this.fluid.config;
    if(!this.playbackSlots)this.playbackSlots=[0,1].map(()=>{
      const texture=d.createTexture({label:'Recorded water volume',size:[cfg.nx,cfg.ny,cfg.nz],dimension:'3d',format:'r8unorm',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
      return{texture,view:texture.createView({dimension:'3d'})};
    });
    const slots=this.playbackSlots;
    const upload=(index:number,frame:PlaybackFrame)=>{
      // writeTexture accepts tightly packed rows; unlike buffer-to-texture
      // copies, this API does not require 256-byte row alignment.
      d.queue.writeTexture({texture:slots[index].texture},frame.phi as Uint8Array<ArrayBuffer>,{bytesPerRow:cfg.nx,rowsPerImage:cfg.ny},[cfg.nx,cfg.ny,cfg.nz]);
      slots[index].frame=frame;
    };
    let first=slots.findIndex(slot=>slot.frame===a);
    if(first<0){first=slots[0].frame===b?1:0;upload(first,a);}
    let second=slots.findIndex(slot=>slot.frame===b);
    if(second<0){second=1-first;upload(second,b);}
    const pair=`${first}:${second}`;
    if(this.playbackPair!==pair){this.renderGroup=this.makeRenderGroup(slots[first].view,slots[second].view);this.playbackPair=pair;}
    this.replaying=true;this.replayAlpha=Math.max(0,Math.min(1,alpha));this.volumeDirty=false;this.dirty=true;
  }
  settings(field:number,opacity:number){this.field=field;this.opacity=opacity;this.dirty=true;}
  resetCamera(){this.yaw=this.options?.cameraYaw??2.35;this.pitch=this.options?.cameraPitch??.57;this.distance=this.options?.cameraDistance??670;this.center=[...(this.options?.cameraCenter??[256,80,26])] as V3;this.dirty=true;}
  private basis(){
    const cp=Math.cos(this.pitch);
    const eye:V3=[this.center[0]+this.distance*Math.sin(this.yaw)*cp,this.center[1]+this.distance*Math.cos(this.yaw)*cp,this.center[2]+this.distance*Math.sin(this.pitch)];
    const forward=unit(this.center.map((v,i)=>v-eye[i]) as V3);
    const right=unit(cross(forward,[0,0,1]));const up=cross(right,forward);
    return{eye,forward,right,up};
  }
  private controls(){
    const c=this.canvas,o={signal:this.abort.signal};
    c.addEventListener('contextmenu',e=>e.preventDefault(),o);
    c.addEventListener('pointerdown',e=>{
      if(this.pointer)return;c.focus();c.setPointerCapture(e.pointerId);
      this.pointer={id:e.pointerId,x:e.clientX,y:e.clientY,pan:e.button===2||e.button===1||e.shiftKey};this.dirty=true;
    },o);
    c.addEventListener('pointermove',e=>{
      const p=this.pointer;if(!p||p.id!==e.pointerId)return;
      const dx=e.clientX-p.x,dy=e.clientY-p.y;p.x=e.clientX;p.y=e.clientY;
      if(p.pan){const {right,up}=this.basis();const factor=this.distance*.75/Math.max(c.clientHeight,100);this.center=this.center.map((v,i)=>v-dx*right[i]*factor+dy*up[i]*factor) as V3;}
      else{this.yaw+=dx*.006;this.pitch=Math.max(-1.35,Math.min(1.45,this.pitch+dy*.006));}
      this.dirty=true;
    },o);
    const end=(e:PointerEvent)=>{if(this.pointer?.id===e.pointerId){this.pointer=undefined;this.dirty=true;}};
    c.addEventListener('pointerup',end,o);c.addEventListener('pointercancel',end,o);c.addEventListener('lostpointercapture',end,o);
    c.addEventListener('wheel',e=>{e.preventDefault();this.distance=Math.max(130,Math.min(2000,this.distance*Math.exp(e.deltaY*.001)));this.dirty=true;},{...o,passive:false});
    c.addEventListener('dblclick',()=>this.resetCamera(),o);
    c.addEventListener('keydown',e=>{if(e.key==='0'||e.key==='Home'){e.preventDefault();this.resetCamera();}},o);
  }
  private tick=(time:number)=>{
    if(this.disposed)return;
    this.frame=requestAnimationFrame(this.tick);
    if(!this.dirty||this.inFlight||time-this.lastFrame<1000/(this.options?.targetFps??60)-1)return;
    this.lastFrame=time;this.dirty=false;
    try{this.draw();}catch(e){this.onError(e);}
  };
  private draw(){
    const c=this.canvas,d=this.fluid.device;
    const ratio=Math.min(window.devicePixelRatio,1.25)*(this.pointer?.65:1);
    const budget=Math.min(1,Math.sqrt((this.options?.pixelBudget??750000)/Math.max(1,c.clientWidth*c.clientHeight*ratio*ratio)));
    const w=Math.max(2,Math.round(c.clientWidth*ratio*budget)),h=Math.max(2,Math.round(c.clientHeight*ratio*budget));
    if(c.width!==w||c.height!==h){c.width=w;c.height=h;}
    const b=this.basis(),cfg=this.fluid.config;
    this.data.set([...b.eye,0,...b.forward,Math.tan(Math.PI/8),...b.right,w/h,...b.up,this.replaying?0:this.field,cfg.nx,cfg.ny,cfg.nz,cfg.scale,this.opacity,this.replayAlpha,0,0]);
    d.queue.writeBuffer(this.parameters,0,this.data);
    const encoder=d.createCommandEncoder({label:'Orbitable 3D volume'});
    if(this.volumeDirty){const pass=encoder.beginComputePass();pass.setPipeline(this.compute);pass.setBindGroup(0,this.copyGroup);pass.dispatchWorkgroups(Math.ceil(cfg.nx/4),Math.ceil(cfg.ny/4),Math.ceil(cfg.nz/4));pass.end();this.volumeDirty=false;}
    const pass=encoder.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:.03,g:.05,b:.08,a:1},loadOp:'clear',storeOp:'store'}]});
    pass.setPipeline(this.renderPipeline);pass.setBindGroup(0,this.renderGroup);pass.draw(3);pass.end();
    d.queue.submit([encoder.finish()]);this.frames++;this.inFlight=true;
    void d.queue.onSubmittedWorkDone().catch(e=>{if(!this.disposed)this.onError(e);}).finally(()=>this.inFlight=false);
  }
  destroy(){
    if(this.disposed)return;this.disposed=true;cancelAnimationFrame(this.frame);this.abort.abort();this.resize.disconnect();
    this.captureResources?.packed.destroy();this.captureResources?.staging.destroy();
    this.playbackSlots?.forEach(slot=>slot.texture.destroy());
    this.volume.destroy();this.parameters.destroy();this.particles.destroy();this.particleGroups.destroy();this.context.unconfigure();
  }
}
