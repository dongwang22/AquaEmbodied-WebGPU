import test from 'node:test';
import assert from 'node:assert/strict';
import {robotPose,packRobot,ROBOT_DOMAIN,ROBOT_GEOMETRY_SCALE,ROBOT_MAX_WATER_LEVEL,ROBOT_ROUTINE_SECONDS,ROBOT_TIME_SCALE,capsuleSignedDistance,type Capsule,type V3} from './robot-motion';
import {readRobotParameters,robotFluidConfig,robotTimeStep,robotRefinement} from './robot-config';
import {readFileSync} from 'node:fs';

const body=(pose:Capsule[],name:string)=>{
  const found=pose.find(b=>b.name===name);assert.ok(found,`Missing body ${name}`);return found;
};
const end=(b:Capsule,sign:number):V3=>b.center.map((v,i)=>v+sign*b.halfLength*b.axis[i]) as V3;
const distance=(a:V3,b:V3)=>Math.hypot(...a.map((v,i)=>v-b[i]));

test('robot anatomy remains finite, connected and inside the pool coordinates through all thirty seconds',()=>{
  for(let frame=0;frame<=600;frame++){
    const t=frame*.05;
    const pose=robotPose(t);assert.equal(pose.length,22);
    for(const b of pose){
      assert.ok([...b.center,...b.axis,b.halfLength,b.radius].every(Number.isFinite));
      assert.ok(Math.abs(Math.hypot(...b.axis)-1)<1e-10);
      for(let d=0;d<3;d++){
        const extent=b.radius+Math.abs(b.axis[d])*b.halfLength;
        assert.ok(b.center[d]-extent>.5,`${b.name} crossed lower wall at ${t}`);
        assert.ok(b.center[d]+extent<ROBOT_DOMAIN[d]-.5);
      }
      if(b.name.endsWith('thigh')||b.name.endsWith('shin'))assert.ok(Math.abs(2*b.halfLength-25.5*ROBOT_GEOMETRY_SCALE)<1e-9);
      if(b.name.endsWith('upper arm'))assert.ok(Math.abs(2*b.halfLength-17*ROBOT_GEOMETRY_SCALE)<1e-9);
      if(b.name.endsWith('forearm'))assert.ok(Math.abs(2*b.halfLength-15*ROBOT_GEOMETRY_SCALE)<1e-9);
    }
    for(const side of ['left','right']){
      const thigh=body(pose,`${side} thigh`),shin=body(pose,`${side} shin`),knee=body(pose,`${side} knee`);
      assert.ok(distance(end(thigh,1),end(shin,-1))<1e-9,`${side} knee disconnected at ${t}`);
      assert.ok(distance(end(thigh,1),knee.center)<1e-9,`${side} knee sphere disconnected at ${t}`);
      const arm=body(pose,`${side} upper arm`),forearm=body(pose,`${side} forearm`),elbow=body(pose,`${side} elbow`),hand=body(pose,`${side} hand`);
      assert.ok(distance(end(arm,1),end(forearm,-1))<1e-9,`${side} elbow disconnected at ${t}`);
      assert.ok(distance(end(arm,1),elbow.center)<1e-9,`${side} elbow sphere disconnected at ${t}`);
      assert.ok(distance(end(forearm,1),hand.center)<1e-9,`${side} hand disconnected at ${t}`);
    }
  }
});
test('initial feet are submerged and water reaches the knees',()=>{
  const pose=robotPose(0),knee=pose.find(b=>b.name==='left knee')!,foot=pose.find(b=>b.name==='left foot')!;
  assert.ok(Math.abs(knee.center[2]-15)<2);
  assert.ok(foot.center[2]+foot.radius<15);
});
test('packed wall velocities match time derivatives of material capsule points',()=>{
  const t=2.35,dt=.001,scale=.75,h=.00001;
  const a=robotPose(t-h),b=robotPose(t+h),p=packRobot(t,scale,dt);
  for(let i=0;i<a.length;i++)for(let d=0;d<3;d++){
    const derivative=(b[i].center[d]-a[i].center[d])/(2*h)*scale*dt;
    assert.ok(Math.abs(p[i*16+4+d]-derivative)<1e-6);
  }
  const first=packRobot(0,scale,dt);
  for(let i=0;i<first.length;i+=16)for(const d of [4,5,6,8,9,10])assert.ok(Math.abs(first[i+d])<1e-8,'startup must be at rest');
});
test('stationary control has identical geometry and zero wall velocity at all times',()=>{
  assert.deepEqual(packRobot(0,1,.001,true),packRobot(8,1,.001,true));
});
test('motion speed rescales the same choreography and the complete rigid wall velocity',()=>{
  const referenceTime=14.3,scale=.75,dt=.0002,reference=packRobot(referenceTime,scale,dt);
  for(const speed of [.5,1,2.5,4]){
    assert.deepEqual(robotPose(referenceTime/speed,speed),robotPose(referenceTime));
    const sped=packRobot(referenceTime/speed,scale,dt,false,speed);
    for(let i=0;i<reference.length;i+=16){
      for(const d of [0,1,2,3,7,11,12,13,14,15])assert.ok(Math.abs(sped[i+d]-reference[i+d])<1e-6);
      for(const d of [4,5,6,8,9,10])assert.ok(Math.abs(sped[i+d]-speed*reference[i+d])<2e-6,`velocity ${i+d} does not follow motion speed`);
    }
  }
});
test('final standing pose is held after the choreography ends with zero moving-wall velocity',()=>{
  const final=robotPose(30);
  assert.deepEqual(robotPose(31),final);assert.deepEqual(robotPose(120),final);
  for(const speed of [.5,1,2.5,4]){
    const packed=packRobot(30/speed+.01,.75,.00025,false,speed);
    for(let i=0;i<packed.length;i+=16)for(const d of [4,5,6,8,9,10])assert.ok(Math.abs(packed[i+d])<1e-12,'completed choreography must not keep moving');
  }
  assert.ok(body(final,'head').center[2]>80*ROBOT_GEOMETRY_SCALE,'head should finish above the standing torso');
  for(const side of ['left','right']){
    const foot=body(final,`${side} foot`),hand=body(final,`${side} hand`);
    assert.ok(foot.center[2]-foot.radius<3*ROBOT_GEOMETRY_SCALE,`${side} foot must finish on the floor`);
    assert.ok(hand.center[2]>body(final,'shoulders').center[2],`${side} arm must finish raised`);
  }
});
test('major choreography transitions do not jump positions or velocities',()=>{
  const h=1e-4;
  for(const t of [5,12,20,26,30]){
    const previous=robotPose(t-h),current=robotPose(t),next=robotPose(t+h);
    for(let i=0;i<current.length;i++){
      assert.ok(distance(previous[i].center,next[i].center)<.15,`${current[i].name} position jumps at ${t}`);
      const dv=current[i].center.map((v,d)=>(next[i].center[d]-2*v+previous[i].center[d])/h);
      assert.ok(Math.hypot(...dv)<.3,`${current[i].name} velocity jumps at ${t}`);
    }
  }
});
test('angular wall velocity agrees with derivatives of capsule endpoints during articulated motion',()=>{
  const t=15.37,dt=.00025,scale=.75,h=.00001,a=robotPose(t-h),b=robotPose(t+h),p=packRobot(t,scale,dt);
  for(let i=0;i<a.length;i++){
    const base=i*16,axis=[p[base+12],p[base+13],p[base+14]],omega=[p[base+8],p[base+9],p[base+10]],r=axis.map(x=>x*p[base+7]);
    const cross=[omega[1]*r[2]-omega[2]*r[1],omega[2]*r[0]-omega[0]*r[2],omega[0]*r[1]-omega[1]*r[0]];
    for(let d=0;d<3;d++){
      const expected=(end(b[i],1)[d]-end(a[i],1)[d])/(2*h)*scale*dt;
      assert.ok(Math.abs(p[base+4+d]+cross[d]-expected)<2e-6,`${a[i].name} endpoint wall velocity is inconsistent`);
    }
  }
});
test('capsule collision geometry matches endpoints and side surface',()=>{
  const b={name:'test',center:[0,0,0] as [number,number,number],axis:[0,0,1] as [number,number,number],halfLength:5,radius:2,material:0};
  assert.equal(capsuleSignedDistance([0,0,7],b),0);
  assert.equal(capsuleSignedDistance([2,0,3],b),0);
  assert.equal(capsuleSignedDistance([0,0,0],b),-2);
  assert.ok(capsuleSignedDistance([3,0,0],b)>0);
});
test('refinement preserves prescribed world motion and gravity time mapping',()=>{
  const raw=JSON.parse(readFileSync(new URL('../public/robotInputParameter.json',import.meta.url),'utf8'));
  const fine=readRobotParameters({...raw,gridScale:.5625}),coarse=readRobotParameters({...raw,gridScale:.375});
  const a=packRobot(2,.5625,robotTimeStep(fine)),b=packRobot(2,.375,robotTimeStep(coarse));
  for(let i=0;i<a.length;i+=16)for(let d=0;d<4;d++)assert.ok(Math.abs(a[i+d]-1.5*b[i+d])<1e-5);
  assert.equal(robotFluidConfig(coarse).gravity*.375,robotFluidConfig(fine).gravity*.5625);
  assert.throws(()=>readRobotParameters({...raw,stepsPerBatch:0}));
  assert.throws(()=>readRobotParameters({...raw,secondsPerStep:NaN}));
});
test('temporal refinement preserves reference fluid coefficients across motion speeds and grids',()=>{
  const raw=JSON.parse(readFileSync(new URL('../public/robotInputParameter.json',import.meta.url),'utf8'));
  const near=(actual:number,expected:number)=>assert.ok(Math.abs(actual-expected)<1e-12*Math.max(1,Math.abs(expected)));
  for(const scale of [.375,.4375,.5625])for(const speed of [.5,1,2.5,4])for(const refinement of [4,8,32]){
    const p=readRobotParameters({...raw,gridScale:scale,motionSpeed:speed,timeRefinement:refinement});
    const q=refinement*Math.max(1,speed*ROBOT_TIME_SCALE),dt=robotTimeStep(p),fluid=robotFluidConfig(p);
    assert.equal(robotRefinement(p),q);
    near(dt*scale*q,p.secondsPerStep);
    near(fluid.nu*q/scale,p.viscosity);
    near(fluid.sigma*q*q/scale,p.surfaceTension);
    near(fluid.gravity*scale*q*q,p.gravity);
  }
});
test('legacy robot parameters default to four temporal subdivisions and reject unsafe refinements',()=>{
  const raw=JSON.parse(readFileSync(new URL('../public/robotInputParameter.json',import.meta.url),'utf8'));
  delete raw.timeRefinement;
  assert.equal(readRobotParameters(raw).timeRefinement,4);
  for(const timeRefinement of [0,3,4.5,33,NaN,Infinity])assert.throws(()=>readRobotParameters({...raw,timeRefinement}));
});
test('adaptive temporal refinement keeps the full capsule wall speed below the LBM stability envelope',()=>{
  const raw=JSON.parse(readFileSync(new URL('../public/robotInputParameter.json',import.meta.url),'utf8'));
  for(const speed of [.5,1,2.5,4]){
    const p=readRobotParameters({...raw,motionSpeed:speed,timeRefinement:4});
    for(let frame=0;frame<=1000;frame++){
      const physicalTime=frame*.01,packed=packRobot(physicalTime/speed,p.gridScale,robotTimeStep(p),false,speed*ROBOT_TIME_SCALE);
      for(let i=0;i<packed.length;i+=16){
        // This upper bound includes both translating and rotating capsule surfaces.
        const speedBound=Math.hypot(packed[i+4],packed[i+5],packed[i+6])+Math.hypot(packed[i+8],packed[i+9],packed[i+10])*(packed[i+3]+packed[i+7]);
        assert.ok(Number.isFinite(speedBound)&&speedBound<.2,`body ${i/16} exceeds safe wall motion at ${physicalTime}s and ${speed}x`);
      }
    }
  }
});
test('routine 5 high-water transition keeps every moving wall below the LBM stability envelope',()=>{
  const raw=JSON.parse(readFileSync(new URL('../public/robotInputParameter.json',import.meta.url),'utf8'));
  for(const waterLevel of [20,25,ROBOT_MAX_WATER_LEVEL]){
    const p=readRobotParameters({...raw,waterLevel,routine:5,motionSpeed:1,timeRefinement:4});
    for(let frame=0;frame<=1000;frame++){
      const phaseTime=(23+frame*.001)/ROBOT_TIME_SCALE,packed=packRobot(phaseTime,p.gridScale,robotTimeStep(p),false,ROBOT_TIME_SCALE,5,waterLevel);
      for(let i=0;i<packed.length;i+=16){
        const speedBound=Math.hypot(packed[i+4],packed[i+5],packed[i+6])+Math.hypot(packed[i+8],packed[i+9],packed[i+10])*(packed[i+3]+packed[i+7]);
        assert.ok(Number.isFinite(speedBound)&&speedBound<.2,`body ${i/16} exceeds safe wall motion at ${phaseTime}s, water ${waterLevel}`);
      }
    }
  }
});

test('the UI choreography maps every complete routine proportionally to ten seconds',()=>{
  assert.equal(ROBOT_ROUTINE_SECONDS,10);
  for(const routine of [1,2,3,4,5,6] as const)
    assert.deepEqual(robotPose(ROBOT_ROUTINE_SECONDS,ROBOT_TIME_SCALE,routine),robotPose(30,1,routine));
});

