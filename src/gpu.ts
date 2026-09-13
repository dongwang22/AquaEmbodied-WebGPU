import common from './shaders/common.wgsl?raw';
import geometry from './shaders/geometry.wgsl?raw';
import curvature from './shaders/curvature.wgsl?raw';
import fluid from './shaders/fluid.wgsl?raw';
import readback from './shaders/readback.wgsl?raw';
import { validateFluidParameters } from './parameters';
import { populationSource, type PopulationPrecision } from './population-storage';
import {FLUID_FACES,fluidBoundaryConditions} from './boundary-topology';
export interface GPUOptions {precision?:PopulationPrecision|'auto';optimized?:boolean;}

export interface FluidConfig {
  nx: number; ny: number; nz: number; scale: number;
  nu: number; sigma: number; gravity: number; particleRadius: number;
  /** Contact angles in degrees. Gravity is the signed z acceleration. */
  thetaP: number; thetaS: number;
  /** Dimensionless Smagorinsky constant; zero disables LES. Delta=1 lattice cell. */
  smagorinsky: number;
  /** This standalone solver only supports the prescribed-motion robot pool. */
  scene: 'robot-pool';
  waterLevel: number;
}
export interface FluidSnapshot {
  rho: Float32Array; phi: Float32Array;
  /** AoS: [ux0, uy0, uz0, ux1, uy1, uz1, ...]. */
  velocity: Float32Array; flags: Uint32Array;
}
export interface FluidDiagnostics {
  step: number; mass: number; pendingMass: number; trappedMass: number;
  conservedMass: number; volume: number; minRho: number; maxRho: number;
  maxSpeed: number; invalidCells: number;
  fluidCells: number; interfaceCells: number; gasCells: number; solidCells: number;
  memoryBytes: number; adapter: string;
  lesCoefficient: number; maxEddyViscosity: number; meanEddyViscosity: number; maxEffectiveViscosity: number;
}

type Entry = 'initialize' | 'snapshotFlags' | 'mapParticles' | 'updateWalls' | 'repairInterface'
  | 'mapAndUpdateWalls' | 'refillUncovered'
  | 'countMassRecipients' | 'distributeMass' | 'clearDistributedMass'
  | 'surfaceExchange' | 'collide' | 'surfacePromote' | 'surfaceCreate' | 'surfaceCommit'
  | 'gatherSlice' | 'gatherAll' | 'gatherBoundary' | 'reduceDiagnostics';

/** 3-D free-surface D3Q19 LBM. In-place Esoteric-Pull populations are SoA;
 * particles are packed 12 floats each, in lattice coordinates. Browser
 * storage bindings are sized from the adapter, never assumed to be 2 GB.
 */
export class FluidGPU {
  readonly precision:PopulationPrecision;
  readonly optimized:boolean;
  readonly cellCount: number;
  readonly memoryBytes: number;
  readonly adapterDescription: string;
  readonly populationBuffers: GPUBuffer[] = [];
  readonly cellBuffer: GPUBuffer;
  private readonly flagsBuffer: GPUBuffer;
  private readonly parameterBuffer: GPUBuffer;
  private readonly particleBuffer: GPUBuffer;
  private readonly sliceBuffer: GPUBuffer;
  private readonly sliceReadback: GPUBuffer;
  private readonly diagnosticBuffer: GPUBuffer;
  private readonly diagnosticReadback: GPUBuffer;
  private readonly pipelines = new Map<Entry, GPUComputePipeline>();
  private readonly groups = new Map<Entry, GPUBindGroup>();
  private readonly planeGroups: Array<{ start: number; count: number }> = [];
  private readonly owned: GPUBuffer[] = [];
  private readonly uniformData = new ArrayBuffer(96);
  readonly particleCount: number;
  private pending: Promise<unknown> = Promise.resolve();
  private failure: string | undefined;
  private disposed = false;
  private stepIndex = 0;
  private previousBounds:{low:number[];high:number[]}|undefined;
  private stepPass:GPUComputePassEncoder|undefined;
  private kinematicFrames:GPUBuffer|undefined;
  private readonly bodyStride=16;

  private constructor(readonly device: GPUDevice, readonly config: FluidConfig,
    adapter: GPUAdapter, particles: Float32Array, planesPerBuffer: number,precision:PopulationPrecision,optimized:boolean) {
    this.precision=precision;this.optimized=optimized;
    this.cellCount = config.nx * config.ny * config.nz;
    this.particleCount = particles.length / this.bodyStride;
    const info = adapter.info;
    this.adapterDescription = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(' / ') || 'WebGPU adapter';
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const staging = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
    this.cellBuffer = this.buffer('Cell fields (48 bytes/cell)', 48 * this.cellCount, storage);
    this.flagsBuffer = this.buffer('Race-free flag snapshot', 4 * this.cellCount, storage);
    this.parameterBuffer = this.buffer('Lattice parameters', 96, GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.particleBuffer = this.buffer('Particles: position, velocity, omega', Math.max(48, particles.byteLength), storage);
    this.sliceBuffer = this.buffer('Slice pack', config.nx * config.nz * 32, storage);
    this.sliceReadback = this.buffer('Slice readback', this.sliceBuffer.size, staging);
    this.diagnosticBuffer = this.buffer('Per-workgroup diagnostics', Math.ceil(this.cellCount / 64) * 64, storage);
    this.diagnosticReadback = this.buffer('Diagnostic readback', this.diagnosticBuffer.size, staging);
    for (let start = 0; start < 19; start += planesPerBuffer) {
      const count = Math.min(19 - start, planesPerBuffer);
      this.planeGroups.push({ start, count });
      this.populationBuffers.push(this.buffer(`${precision} shifted populations ${start}..${start + count - 1}`, count * this.cellCount * (precision==='fp16s'?2:4), storage));
    }
    this.memoryBytes = this.owned.reduce((sum, buffer) => sum + buffer.size, 0);
    this.upload(particles);
    device.addEventListener('uncapturederror', (event: GPUUncapturedErrorEvent) => { this.failure = event.error.message; });
    void device.lost.then(info => { if (!this.disposed) this.failure = `WebGPU device lost: ${info.message || info.reason}`; });
  }

  static async create(config: FluidConfig, particles: Float32Array, options:GPUOptions={}): Promise<FluidGPU> {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable. Use a compatible Chrome/Edge browser on localhost or HTTPS.');
    if (![config.nx, config.ny, config.nz].every(n => Number.isInteger(n) && n >= 4)) throw new Error('Grid dimensions must be integers >= 4.');
    if(config.scene!=='robot-pool')throw new Error('The standalone build only supports the robot pool scene.');
    if(particles.length%16||particles.length<16)throw new Error('Expected prescribed robot capsules packed as 16 floats each.');
    if(!Number.isFinite(config.waterLevel)||config.waterLevel*config.scale<2||config.waterLevel*config.scale>=config.nz-2)throw new Error('Pool water level must be inside the domain.');
    const {scene: _scene,waterLevel:_waterLevel,...numericConfig}=config;
    validateFluidParameters(numericConfig);
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found. Check browser hardware acceleration.');
    const requested=options.precision??'fp32';
    if(!['auto','fp32','fp16s'].includes(requested))throw new Error('Population precision must be fp32, fp16s or auto.');
    if(requested==='fp16s'&&!adapter.features.has('shader-f16'))throw new Error('FP16S requires shader-f16 support. Select fp32 in inputParameter.json.');
    const precision:PopulationPrecision=requested==='auto'?(adapter.features.has('shader-f16')?'fp16s':'fp32'):requested;
    const bytesPerValue=precision==='fp16s'?2:4;
    const cells = config.nx * config.ny * config.nz;
    const limit = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
    if (48 * cells > limit) throw new Error(`This grid needs ${(48*cells/1048576).toFixed(0)} MiB per cell buffer; the GPU limit is ${(limit/1048576).toFixed(0)} MiB. Reduce gridScale in inputParameter.json.`);
    const planes = Math.min(19, Math.floor(limit / (cells * bytesPerValue)));
    const popBindings = Math.ceil(19 / planes);
    const storageBindings = popBindings + 3;
    if (storageBindings > adapter.limits.maxStorageBuffersPerShaderStage) throw new Error(`This grid needs ${storageBindings} storage bindings; the GPU supports ${adapter.limits.maxStorageBuffersPerShaderStage}. Reduce gridScale in inputParameter.json.`);
    const maxBinding = Math.max(48 * cells, Math.min(planes, 19) * cells * bytesPerValue, Math.ceil(cells / 64) * 64);
    const device = await adapter.requestDevice({requiredFeatures:precision==='fp16s'?['shader-f16']:[], requiredLimits: {
      maxStorageBufferBindingSize: maxBinding,
      maxBufferSize: maxBinding,
      maxStorageBuffersPerShaderStage: Math.max(8, storageBindings),
      maxBindingsPerBindGroup: Math.max(1000, 6 + popBindings),
    }});
    const result = new FluidGPU(device, { ...config }, adapter, particles, planes,precision,options.optimized??true);
    try {
      await result.compile();
      const encoder = device.createCommandEncoder({ label: 'Initialize robot water pool' });
      result.dispatch(encoder, 'initialize');
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      result.assertHealthy();
      return result;
    } catch (error) { result.destroy(); throw error; }
  }

  private buffer(label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    const buffer = this.device.createBuffer({ label, size: Math.ceil(size / 4) * 4, usage });
    this.owned.push(buffer); return buffer;
  }

  private populationSource(): string {
    return populationSource(this.planeGroups,this.precision);
  }

  private async compile(): Promise<void> {
    const bodyCommon=common.replace('/* BODY_AXIS */','axis:vec4<f32>,')
      .replace('/* BODY_DISTANCE */','r-=body.axis.xyz*clamp(dot(r,body.axis.xyz),-body.velocity.w,body.velocity.w);');
    const sceneCode=`const POOL_LEVEL:f32=${Number(this.config.waterLevel).toFixed(8)};`;
    const code = [this.precision==='fp16s'?'enable f16;':'',sceneCode,bodyCommon, this.populationSource(), geometry, curvature, fluid, readback].join('\n');
    const module = this.device.createShaderModule({ label: 'Robot pool 3D free-surface LBM', code });
    const messages = await module.getCompilationInfo();
    const errors = messages.messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(e => `WGSL ${e.lineNum}:${e.linePos} ${e.message}`).join('\n'));
    const pop = this.populationBuffers.map((_, i) => i + 6);
    const bindings: Record<Entry, number[]> = {
      initialize: [0, 1, 2, 3, ...pop], snapshotFlags: [0, 1, 2],
      mapParticles: [0, 1, 3], updateWalls: [0, 1, 3], repairInterface: [0, 1, 2, ...pop],
      refillUncovered: [0, 1, 2, ...pop],
      mapAndUpdateWalls:[0,1,2,3],
      countMassRecipients: [0, 1, 2], distributeMass: [0, 1, 2], clearDistributedMass: [0, 1],
      surfaceExchange: [0, 1, 2, ...pop], collide: [0, 1, 2, ...pop],
      surfacePromote: [0, 1, 2], surfaceCreate: [0, 1, 2, ...pop], surfaceCommit: [0, 1, 2],
      gatherSlice: [0, 1, 5], gatherAll: [0, 1, 5], gatherBoundary: [0, 1, 5], reduceDiagnostics: [0, 1, 5],
    };
    const resources: GPUBuffer[] = [this.parameterBuffer, this.cellBuffer, this.flagsBuffer,
      this.particleBuffer, this.cellBuffer, this.sliceBuffer, ...this.populationBuffers];
    for (const [entry, used] of Object.entries(bindings) as [Entry, number[]][]) {
      const pipeline = await this.device.createComputePipelineAsync({ label: entry, layout: 'auto', compute: { module, entryPoint: entry,constants:{optimizedKernels:Number(this.optimized)} } });
      this.pipelines.set(entry, pipeline);
      const entries = used.map(binding => ({ binding, resource: { buffer: binding === 5 && entry === 'reduceDiagnostics' ? this.diagnosticBuffer : resources[binding] } }));
      this.groups.set(entry, this.device.createBindGroup({ label: entry, layout: pipeline.getBindGroupLayout(0), entries }));
    }
  }

  private prepareUniform(particles: Float32Array,step=this.stepIndex): void {
    if (particles.length !== this.particleCount * this.bodyStride) throw new Error('Body count cannot change after creation.');
    const u = new Uint32Array(this.uniformData); const f = new Float32Array(this.uniformData); const c = this.config;
    u.set([c.nx, c.ny, c.nz, this.cellCount]);
    f.set([c.nu, c.sigma, c.gravity, c.scale], 4);
    f.set([c.particleRadius, c.thetaP * Math.PI / 180, c.thetaS * Math.PI / 180, c.smagorinsky], 8);
    const low = [Infinity, Infinity, Infinity]; const high = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < particles.length; i += this.bodyStride) {
      for (let j = 0; j < this.bodyStride; j++) if (!Number.isFinite(particles[i + j])) throw new Error('Body state contains NaN/Infinity.');
      const radius=particles[i+3]+(this.bodyStride===16?particles[i+7]:0);
      for (let d = 0; d < 3; d++) { low[d] = Math.min(low[d], particles[i + d] - radius); high[d] = Math.max(high[d], particles[i + d] + radius); }
    }
    const previous=this.previousBounds;this.previousBounds={low:[...low],high:[...high]};
    // Union of old and new particle envelopes covers all changed cells, even a
    // large clipped displacement. Two extra cells suffice for interface repair.
    f.set([...low.map((v,i)=>Math.min(v,previous?.low[i]??v)), 0], 12);
    f.set([...high.map((v,i)=>Math.max(v,previous?.high[i]??v)), 0], 16);
    u.set([step, this.particleCount, Math.floor(c.ny / 2), c.scale===1?90:30], 20);
  }
  private upload(particles:Float32Array):void{
    this.prepareUniform(particles);
    this.device.queue.writeBuffer(this.parameterBuffer, 0, this.uniformData);
    this.device.queue.writeBuffer(this.particleBuffer, 0, particles as Float32Array<ArrayBuffer>);
  }

  /** Prescribed robot bodies drive the fluid with no dynamics feedback.
   * Each lattice step receives its OWN pose through ordered GPU copies. */
  advanceKinematic(poses:readonly Float32Array[]):Promise<void>{
    return this.enqueue(async()=>{
      if(!poses.length||poses.length>128)throw new Error('Supply 1–128 successive lattice-step poses.');
      const bytes=this.particleCount*64,slot=256+bytes;
      this.kinematicFrames??=this.buffer('Ordered prescribed poses',slot*128,GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
      const data=new Uint8Array(slot*poses.length);
      for(let k=0;k<poses.length;k++){
        this.prepareUniform(poses[k],this.stepIndex+k);
        data.set(new Uint8Array(this.uniformData),k*slot);
        data.set(new Uint8Array(poses[k].buffer,poses[k].byteOffset,bytes),k*slot+256);
      }
      this.device.queue.writeBuffer(this.kinematicFrames,0,data);
      const encoder=this.device.createCommandEncoder({label:'Prescribed robot / LBM–VOF'});
      for(let k=0;k<poses.length;k++){
        encoder.copyBufferToBuffer(this.kinematicFrames,k*slot,this.parameterBuffer,0,96);
        encoder.copyBufferToBuffer(this.kinematicFrames,k*slot+256,this.particleBuffer,0,bytes);
        this.stepPass=encoder.beginComputePass({label:'One-way fluid stages'});
        this.dispatch(encoder,'mapAndUpdateWalls');
        this.dispatch(encoder,'refillUncovered');this.dispatch(encoder,'snapshotFlags');this.dispatch(encoder,'repairInterface');
        if(!this.optimized)this.dispatch(encoder,'snapshotFlags');
        this.dispatch(encoder,'countMassRecipients');this.dispatch(encoder,'distributeMass');
        if(!this.optimized)this.dispatch(encoder,'clearDistributedMass');
        this.dispatch(encoder,'surfaceExchange');this.dispatch(encoder,'collide');
        this.dispatch(encoder,'snapshotFlags');this.dispatch(encoder,'surfacePromote');
        this.dispatch(encoder,'snapshotFlags');this.dispatch(encoder,'surfaceCreate');
        this.dispatch(encoder,'snapshotFlags');this.dispatch(encoder,'surfaceCommit');
        this.stepPass.end();this.stepPass=undefined;
      }
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      // Only expose a pose/time whose corresponding water step has completed.
      this.stepIndex+=poses.length;this.assertHealthy();
    });
  }

  private dispatch(encoder: GPUCommandEncoder, entry: Entry, count = this.cellCount, groupOverride?: GPUBindGroup): void {
    const pass = this.stepPass??encoder.beginComputePass({ label: entry });
    pass.setPipeline(this.pipelines.get(entry)!); pass.setBindGroup(0, groupOverride ?? this.groups.get(entry)!);
    const groups = Math.ceil(count / 64);
    pass.dispatchWorkgroups(Math.min(65535, groups), Math.ceil(groups / 65535));
    if(!this.stepPass)pass.end();
  }

  private assertHealthy(): void {
    if (this.disposed) throw new Error('FluidGPU has been destroyed.');
    if (this.failure) throw new Error(this.failure);
  }

  private enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const next = this.pending.then(() => { this.assertHealthy(); return operation(); });
    this.pending = next.catch(() => undefined); return next;
  }

  slice(y: number): Promise<FluidSnapshot> {
    return this.enqueue(async () => {
      new Uint32Array(this.uniformData)[22] = Math.max(0, Math.min(this.config.ny - 1, Math.floor(y)));
      this.device.queue.writeBuffer(this.parameterBuffer, 88, new Uint32Array(this.uniformData,88,1));
      const encoder = this.device.createCommandEncoder({ label: 'Gather cross section' });
      this.dispatch(encoder, 'gatherSlice', this.config.nx * this.config.nz);
      encoder.copyBufferToBuffer(this.sliceBuffer, 0, this.sliceReadback, 0, this.sliceBuffer.size);
      this.device.queue.submit([encoder.finish()]);
      return this.unpack(await this.read(this.sliceReadback));
    });
  }

  snapshot(): Promise<FluidSnapshot> {
    return this.enqueue(async () => {
      const size = this.cellCount * 32;
      const output = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        const group = this.device.createBindGroup({ layout: this.pipelines.get('gatherAll')!.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: this.parameterBuffer } }, { binding: 1, resource: { buffer: this.cellBuffer } }, { binding: 5, resource: { buffer: output } },
        ] });
        const encoder = this.device.createCommandEncoder(); this.dispatch(encoder, 'gatherAll', this.cellCount, group);
        encoder.copyBufferToBuffer(output, 0, staging, 0, size); this.device.queue.submit([encoder.finish()]);
        return this.unpack(await this.read(staging));
      } finally { output.destroy(); staging.destroy(); }
    });
  }

  /** Read actual GPU cells on all six faces without downloading the full volume. */
  boundaries() {
    return this.enqueue(async () => {
      const {nx,ny,nz}=this.config, counts=[ny*nz,ny*nz,nx*nz,nx*nz,nx*ny,nx*ny];
      const count=counts.reduce((a,b)=>a+b,0), size=count*32;
      const output=this.device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
      const staging=this.device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
      try {
        const group=this.device.createBindGroup({layout:this.pipelines.get('gatherBoundary')!.getBindGroupLayout(0),entries:[
          {binding:0,resource:{buffer:this.parameterBuffer}},{binding:1,resource:{buffer:this.cellBuffer}},{binding:5,resource:{buffer:output}},
        ]});
        const encoder=this.device.createCommandEncoder();this.dispatch(encoder,'gatherBoundary',count,group);
        encoder.copyBufferToBuffer(output,0,staging,0,size);this.device.queue.submit([encoder.finish()]);
        const data=await this.read(staging), floats=new Float32Array(data), bits=new Uint32Array(data);
        const conditions=fluidBoundaryConditions();
        let offset=0;
        const faces=counts.map((count,face)=>{
          let violations=0,maxSpeed=0,maxMass=0;
          for(let i=0;i<count;i++) {
            const j=8*(offset+i),speed=Math.hypot(floats[j+4],floats[j+5],floats[j+6]);
            maxSpeed=Math.max(maxSpeed,speed);maxMass=Math.max(maxMass,Math.abs(floats[j+2]));
            const z=face<2?Math.floor(i/ny):face<4?Math.floor(i/nx):(face===4?0:nz-1);
            const expectedSolid=face>=4||z===0||z===nz-1;
            if(expectedSolid){
              if(bits[j]!==1||bits[j+1]!==0||floats[j+2]!==0||floats[j+3]!==0||speed!==0)violations++;
            }else if(bits[j]===1){
              // Interior rows of the x/y edge planes are active periodic cells.
              violations++;
            }
          }
          offset+=count;return{face:FLUID_FACES[face],condition:conditions[face],count,violations,maxSpeed,maxMass};
        });
        const valid=faces.every(f=>f.violations===0);
        this.assertHealthy();return{
          step:this.stepIndex,valid,
          closed:false,
          periodicSides:valid&&conditions.slice(0,4).every(condition=>condition==='periodic'),
          solidTopBottom:valid&&conditions.slice(4).every(condition=>condition==='solid-wall'),
          faces,
        };
      }finally{output.destroy();staging.destroy();}
    });
  }

  diagnostics(): Promise<FluidDiagnostics> {
    return this.enqueue(async () => {
      const encoder = this.device.createCommandEncoder({ label: 'Check conservation and stability' });
      this.dispatch(encoder, 'reduceDiagnostics');
      encoder.copyBufferToBuffer(this.diagnosticBuffer, 0, this.diagnosticReadback, 0, this.diagnosticBuffer.size);
      this.device.queue.submit([encoder.finish()]);
      const a = new Float32Array(await this.read(this.diagnosticReadback));
      const result: FluidDiagnostics = { step: this.stepIndex, mass: 0, pendingMass: 0, trappedMass: 0, conservedMass: 0, volume: 0,
        minRho: Infinity, maxRho: -Infinity, maxSpeed: 0, invalidCells: 0, fluidCells: 0, interfaceCells: 0, gasCells: 0, solidCells: 0,
        memoryBytes: this.memoryBytes, adapter: this.adapterDescription,
        lesCoefficient:this.config.smagorinsky,maxEddyViscosity:0,meanEddyViscosity:0,maxEffectiveViscosity:this.config.nu };
      for (let i = 0; i < a.length; i += 16) {
        result.mass += a[i]; result.pendingMass += a[i + 1]; result.trappedMass += a[i + 2]; result.volume += a[i + 3];
        result.minRho = Math.min(result.minRho, a[i + 4]); result.maxRho = Math.max(result.maxRho, a[i + 5]);
        result.maxSpeed = Math.max(result.maxSpeed, a[i + 6]); result.invalidCells += a[i + 7];
        result.fluidCells += a[i + 8]; result.interfaceCells += a[i + 9]; result.gasCells += a[i + 10]; result.solidCells += a[i + 11];
        result.maxEddyViscosity=Math.max(result.maxEddyViscosity,a[i+12]);result.meanEddyViscosity+=a[i+13];
      }
      result.meanEddyViscosity/=Math.max(1,result.fluidCells+result.interfaceCells);
      result.maxEffectiveViscosity=this.config.nu+result.maxEddyViscosity;
      result.conservedMass = result.mass + result.pendingMass + result.trappedMass;
      this.assertHealthy(); return result;
    });
  }

  private async read(buffer: GPUBuffer): Promise<ArrayBuffer> {
    await buffer.mapAsync(GPUMapMode.READ);
    try { return buffer.getMappedRange().slice(0); } finally { buffer.unmap(); }
  }
  private unpack(buffer: ArrayBuffer): FluidSnapshot {
    const src = new Float32Array(buffer); const bits = new Uint32Array(buffer); const count = src.length / 8;
    const rho = new Float32Array(count); const phi = new Float32Array(count); const velocity = new Float32Array(count * 3); const flags = new Uint32Array(count);
    for (let i = 0; i < count; i++) {
      rho[i] = src[i * 8]; phi[i] = src[i * 8 + 1]; flags[i] = bits[i * 8 + 2];
      velocity[i * 3] = src[i * 8 + 4]; velocity[i * 3 + 1] = src[i * 8 + 5]; velocity[i * 3 + 2] = src[i * 8 + 6];
    }
    return { rho, phi, velocity, flags };
  }
  destroy(): void {
    if (this.disposed) return; this.disposed = true;
    for (const buffer of this.owned) buffer.destroy(); this.device.destroy();
  }
}
