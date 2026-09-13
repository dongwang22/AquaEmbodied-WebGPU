import test from 'node:test';
import assert from 'node:assert/strict';
import {robotPose,packRobot,ROBOT_CENTER,ROBOT_DOMAIN,ROBOT_GEOMETRY_SCALE,ROBOT_MAX_WATER_LEVEL,type Capsule,type RoutineId,type V3} from './robot-motion';

const ids:RoutineId[]=[1,2,3,4,5,6];
const body=(pose:Capsule[],name:string):Capsule=>{
  const result=pose.find(b=>b.name===name);assert.ok(result,`Missing ${name}`);return result;
};
const end=(b:Capsule,s:number):V3=>b.center.map((v,d)=>v+s*b.halfLength*b.axis[d]) as V3;
const distance=(a:V3,b:V3)=>Math.hypot(...a.map((v,d)=>v-b[d]));
const floor=(b:Capsule)=>b.center[2]-b.radius-Math.abs(b.axis[2])*b.halfLength;
const near=(a:V3,b:V3,message:string)=>assert.ok(distance(a,b)<1e-8,`${message}: gap ${distance(a,b)}`);
const reach=(pose:Capsule[],side:string)=>distance(end(body(pose,`${side} thigh`),-1),end(body(pose,`${side} shin`),1));
const sub=(a:V3,b:V3):V3=>a.map((v,d)=>v-b[d]) as V3;
const dot=(a:V3,b:V3)=>a.reduce((sum,v,d)=>sum+v*b[d],0);
const clamp=(v:number)=>Math.max(0,Math.min(1,v));
/** Closest points of finite segments, including spheres and parallel bones. */
function segmentDistance(a:Capsule,b:Capsule):number{
  const p=end(a,-1),q=end(b,-1),u=sub(end(a,1),p),v=sub(end(b,1),q),r=sub(p,q);
  const aa=dot(u,u),bb=dot(v,v),uv=dot(u,v),ur=dot(u,r),vr=dot(v,r);
  let s=0,t=0;
  if(aa<1e-15&&bb<1e-15)return distance(p,q);
  if(aa<1e-15)t=clamp(vr/bb);
  else if(bb<1e-15)s=clamp(-ur/aa);
  else{
    const determinant=aa*bb-uv*uv;
    s=determinant>1e-15?clamp((uv*vr-ur*bb)/determinant):0;
    t=(uv*s+vr)/bb;
    if(t<0){t=0;s=clamp(-ur/aa);}
    else if(t>1){t=1;s=clamp((uv-ur)/aa);}
  }
  return Math.hypot(...p.map((x,d)=>x+s*u[d]-q[d]-t*v[d]));
}

test('all six routines keep actual joint endpoints attached to a rigid chassis throughout every phase',()=>{
  for(const routine of ids)for(let frame=0;frame<=600;frame++){
    const t=frame/20,pose=robotPose(t,1,routine),label=`routine ${routine} at ${t}s`;
    assert.equal(pose.length,22);
    for(const b of pose){
      assert.ok([...b.center,...b.axis,b.radius,b.halfLength].every(Number.isFinite),`${label}: ${b.name} nonfinite`);
      assert.ok(Math.abs(Math.hypot(...b.axis)-1)<1e-9,`${label}: ${b.name} axis not normalized`);
      for(let d=0;d<3;d++){
        const extent=b.radius+Math.abs(b.axis[d])*b.halfLength;
        assert.ok(b.center[d]-extent>.5,`${label}: ${b.name} below wall ${d}`);
        assert.ok(b.center[d]+extent<ROBOT_DOMAIN[d]-.5,`${label}: ${b.name} beyond wall ${d}`);
      }
    }
    for(const [side,sign] of [['right',-1],['left',1]] as const){
      const thigh=body(pose,`${side} thigh`),shin=body(pose,`${side} shin`),knee=body(pose,`${side} knee`);
      const upper=body(pose,`${side} upper arm`),forearm=body(pose,`${side} forearm`),elbow=body(pose,`${side} elbow`),hand=body(pose,`${side} hand`);
      for(const [part,length] of [[thigh,25.5],[shin,25.5],[upper,17],[forearm,15]] as const)
        assert.ok(Math.abs(2*part.halfLength-length*ROBOT_GEOMETRY_SCALE)<1e-8,`${label}: ${part.name} changed length`);
      near(end(thigh,-1),end(body(pose,'pelvis'),sign),`${label}: ${side} hip detached`);
      near(end(upper,-1),end(body(pose,'shoulders'),sign),`${label}: ${side} shoulder detached`);
      near(end(thigh,1),end(shin,-1),`${label}: ${side} knee joint detached`);
      near(end(thigh,1),knee.center,`${label}: ${side} knee sphere detached`);
      near(end(upper,1),end(forearm,-1),`${label}: ${side} elbow joint detached`);
      near(end(upper,1),elbow.center,`${label}: ${side} elbow sphere detached`);
      near(end(forearm,1),hand.center,`${label}: ${side} wrist detached`);
    }
  }
});

test('cropped pool contains every routine and preserves a local wave-and-splash margin',()=>{
  const lower=[Infinity,Infinity,Infinity],upper=[Infinity,Infinity,Infinity];
  for(const waterLevel of [4,15,ROBOT_MAX_WATER_LEVEL])for(const routine of ids)for(let frame=0;frame<=600;frame++){
    for(const b of robotPose(frame/20,1,routine,waterLevel))for(let d=0;d<3;d++){
      const extent=b.radius+Math.abs(b.axis[d])*b.halfLength;
      lower[d]=Math.min(lower[d],b.center[d]-extent);
      upper[d]=Math.min(upper[d],ROBOT_DOMAIN[d]-b.center[d]-extent);
    }
  }
  assert.ok(lower[0]>23&&upper[0]>23,'x pool must retain a wave margin around the centered half-size sweep');
  assert.ok(lower[1]>23&&upper[1]>23,'y pool must retain a wave margin around the centered half-size sweep');
  assert.ok(upper[2]>60,'pool must retain splash headroom above the half-size robot');
});

test('standing, floor-support, horse-stance and axial-roll openings have different required body configurations',()=>{
  for(const routine of [1,3,4,6] as RoutineId[])
    assert.ok(body(robotPose(0,1,routine),'torso').axis[2]>.85,`routine ${routine} must start upright`);
  for(const routine of [2,5] as RoutineId[])
    assert.ok(Math.abs(body(robotPose(0,1,routine),'torso').axis[2])<.5,`routine ${routine} must start on the floor`);
  const slap=robotPose(0,1,1),horse=robotPose(0,1,4);
  assert.ok(body(horse,'pelvis').center[2]<body(slap,'pelvis').center[2]-8*ROBOT_GEOMETRY_SCALE,'horse stance must lower the pelvis visibly');
  for(const side of ['right','left']){
    assert.ok(reach(horse,side)<reach(slap,side)-8*ROBOT_GEOMETRY_SCALE,'horse stance must bend the knees, not translate straight legs');
    assert.ok(floor(body(robotPose(0,1,2),`${side} hand`))<3*ROBOT_GEOMETRY_SCALE,'floor-sweep opening must plant both hands');
    assert.ok(body(slap,`${side} hand`).center[2]>body(slap,'shoulders').center[2]-4*ROBOT_GEOMETRY_SCALE,'opening slap arms must start raised to the sides');
  }
});

test('opposite limbs and non-adjacent torso envelopes never intersect at default and extreme water levels',()=>{
  const limbs=['thigh','shin','knee','foot','upper arm','forearm','elbow','hand'];
  for(const waterLevel of [15,4,ROBOT_MAX_WATER_LEVEL])for(const routine of ids)for(let frame=0;frame<=600;frame++){
    const t=frame/20,pose=robotPose(t,1,routine,waterLevel),pairs:[Capsule,Capsule][]=[];
    for(const left of limbs)for(const right of limbs)pairs.push([body(pose,`left ${left}`),body(pose,`right ${right}`)]);
    for(const side of ['left','right']){
      for(const arm of ['upper arm','forearm','elbow','hand'])for(const leg of ['thigh','shin','knee','foot'])
        pairs.push([body(pose,`${side} ${arm}`),body(pose,`${side} ${leg}`)]);
      for(const part of limbs)pairs.push([body(pose,`${side} ${part}`),body(pose,'torso')]);
      pairs.push([body(pose,`${side} hand`),body(pose,'head')]);
    }
    for(const [a,b] of pairs){
      const clearance=segmentDistance(a,b)-a.radius-b.radius;
      assert.ok(clearance>=-.05,`routine ${routine} at ${t}s, water ${waterLevel}: ${a.name} intersects ${b.name} by ${-clearance}`);
    }
  }
});

test('the six 30-second routines have distinguishable complete motion trajectories',()=>{
  const names=['head','left hand','right hand','left foot','right foot'];
  for(let a=1;a<=6;a++)for(let b=a+1;b<=6;b++){
    let squared=0,count=0;
    for(let frame=0;frame<=60;frame++){
      const p=robotPose(frame/2,1,a as RoutineId),q=robotPose(frame/2,1,b as RoutineId);
      for(const name of names){squared+=distance(body(p,name).center,body(q,name).center)**2;count++;}
    }
    assert.ok(Math.sqrt(squared/count)>12*ROBOT_GEOMETRY_SCALE,`routines ${a} and ${b} repeat nearly the same movement`);
  }
});

test('every routine stops at its own finale and motion speed changes time rather than skeletal dimensions',()=>{
  for(const routine of ids){
    const final=robotPose(30,1,routine);
    assert.deepEqual(robotPose(35,1,routine),final);
    for(const speed of [.5,2.5,4]){
      assert.deepEqual(robotPose(11.25/speed,speed,routine),robotPose(11.25,1,routine));
      const packed=packRobot(31/speed,.375,.0002,false,speed,routine);
      for(let i=0;i<packed.length;i+=16)for(const d of [4,5,6,8,9,10])assert.equal(packed[i+d],0);
    }
  }
});

test('all choreography phase handoffs keep positions and wall velocities continuous',()=>{
  const boundaries:Record<RoutineId,number[]>={
    1:[0,5,6,12,13,20,21,30],2:[0,6,7,14,15,22,23,30],3:[0,7,8,14,15,22,23,30],
    4:[0,8,9,16,17,24,25,30],5:[0,7,8,15,16,23,24,30],6:[0,6,7,14,15,22,23,30],
  };
  const h=1e-4;
  for(const routine of ids)for(const t of boundaries[routine]){
    const p=robotPose(t-h,1,routine),q=robotPose(t,1,routine),r=robotPose(t+h,1,routine);
    for(let i=0;i<q.length;i++){
      assert.ok(distance(p[i].center,r[i].center)<.12,`routine ${routine}: ${q[i].name} position jumps at ${t}`);
      const dv=q[i].center.map((v,d)=>(r[i].center[d]-2*v+p[i].center[d])/h);
      assert.ok(Math.hypot(...dv)<.25,`routine ${routine}: ${q[i].name} velocity jumps at ${t}`);
      for(const sign of [-1,1])assert.ok(distance(end(p[i],sign),end(r[i],sign))<.15,`routine ${routine}: ${q[i].name} axis jumps at ${t}`);
    }
  }
});

function samples(routine:RoutineId,start:number,finish:number,read:(pose:Capsule[])=>number):number[]{
  return Array.from({length:1001},(_,i)=>read(robotPose(start+(finish-start)*i/1000,1,routine)));
}
function countExcursions(values:number[],high:boolean,minAmplitude:number):number{
  const low=Math.min(...values),top=Math.max(...values),threshold=(low+top)/2;
  assert.ok(top-low>=minAmplitude*ROBOT_GEOMETRY_SCALE,`motion range ${top-low} is below required ${minAmplitude*ROBOT_GEOMETRY_SCALE}`);
  let count=0,active=false;
  // A tiny deadband avoids counting roundoff at a centered sweep as an extra stroke.
  const deadband=(top-low)*.001;
  for(const value of values){
    const displacement=(high?1:-1)*(value-threshold);
    if(displacement>deadband&&!active){count++;active=true;}
    else if(displacement<-deadband)active=false;
  }
  return count;
}
function strokes(routine:RoutineId,start:number,finish:number,name:string,axis:number,high:boolean,count:number,minAmplitude:number){
  assert.equal(countExcursions(samples(routine,start,finish,p=>body(p,name).center[axis]),high,minAmplitude),count,
    `routine ${routine}, ${start}–${finish}s: ${name} does not perform ${count} strokes`);
}

test('Water Slap Opening performs 3 side slaps, 4 forward slams, 5 scoops and alternating finale strokes',()=>{
  for(const side of ['left','right']){
    strokes(1,0,5,`${side} hand`,2,false,3,35);
    strokes(1,6,12,`${side} hand`,2,false,4,55);
    strokes(1,13,20,`${side} hand`,2,false,5,35);
    strokes(1,21,29.35,`${side} hand`,2,false,5,35);
  }
  for(let t=0;t<=20;t+=.05){
    const pose=robotPose(t),r=body(pose,'right hand'),l=body(pose,'left hand');
    assert.ok(Math.abs(r.center[2]-l.center[2])<1e-8,'paired opening strokes must remain synchronized');
    assert.ok(r.center[1]<ROBOT_CENTER[1]&&l.center[1]>ROBOT_CENTER[1],'side slaps may not cross the body midline');
  }
  const active=samples(1,21,29.35,p=>body(p,'right hand').center[2]-body(p,'left hand').center[2]);
  assert.ok(Math.min(...active)<-25*ROBOT_GEOMETRY_SCALE&&Math.max(...active)>25*ROBOT_GEOMETRY_SCALE,'finale must alternate its active arm');
});

test('Floor Sweep Storm alternates two rear sweeps per leg, then 6 paired kicks, 4 full side cycles and 4 V slaps',()=>{
  strokes(2,0,6,'right foot',1,true,2,35);strokes(2,0,6,'left foot',1,false,2,35);
  for(const side of ['right','left']){
    strokes(2,7,14,`${side} foot`,2,false,6,30);
    strokes(2,15,22,`${side} foot`,1,true,4,60);
    strokes(2,15,22,`${side} foot`,1,false,4,60);
    strokes(2,23,29.4,`${side} foot`,2,false,4,15);
  }
});

test('Standing Kick Surge performs 4 forward kicks per leg, 6 alternating outward kicks and 3 overhead scoops',()=>{
  strokes(3,0,7,'right foot',0,true,4,45);strokes(3,8,14,'left foot',0,true,4,45);
  strokes(3,15,22,'right foot',1,false,3,35);strokes(3,15,22,'left foot',1,true,3,35);
  for(const side of ['right','left'])strokes(3,23,29.3,`${side} hand`,2,false,3,55);
  for(const [start,endTime,support,active] of [[0,7,'left','right'],[8,14,'right','left']] as const){
    const initial=body(robotPose(start,1,3),`${support} foot`).center;
    let peak=0;
    for(let frame=0;frame<=140;frame++){
      const pose=robotPose(start+(endTime-start)*frame/140,1,3),foot=body(pose,`${support} foot`);
      assert.ok(distance(foot.center,initial)<.01,'single-leg support must not float with the kicking leg');
      assert.ok(floor(foot)<3*ROBOT_GEOMETRY_SCALE,'single-leg support must remain on the pool floor');
      peak=Math.max(peak,body(pose,`${active} foot`).center[2]);
    }
    assert.ok(peak>32*ROBOT_GEOMETRY_SCALE,'forward kick must carry the foot through and above the water surface');
  }
});

test('Horse Stance Strikes performs 5 forward chops, 5 side chops, 8 accelerating punches and 3 side slaps',()=>{
  for(const side of ['left','right']){
    strokes(4,0,8,`${side} hand`,2,false,5,45);
    strokes(4,9,16,`${side} hand`,2,false,5,45);
    strokes(4,25,29.35,`${side} hand`,2,false,3,20);
  }
  strokes(4,17,24,'right hand',1,false,4,18);strokes(4,17,24,'left hand',1,true,4,18);
  const values=samples(4,17,24,p=>Math.min(body(p,'right hand').center[2],body(p,'left hand').center[2]));
  assert.ok(Math.min(...values)<30*ROBOT_GEOMETRY_SCALE,'outward punches must enter the selected water surface');
  const threshold=(Math.min(...values)+Math.max(...values))/2,times:number[]=[];
  for(let i=1;i<values.length;i++)if(values[i]<threshold&&values[i-1]>=threshold)times.push(17+7*i/1000);
  assert.equal(times.length,8);
  assert.ok(times[7]-times[6]<.8*(times[1]-times[0]),'the eight punches must accelerate');
});

test('Rolling Sweep Whirlwind completes four axial rolls, five rear slaps, three pendulum slaps per side and three V slaps',()=>{
  const axes=Array.from({length:701},(_,i)=>body(robotPose(i/100,1,5),'shoulders').axis);
  let angle=0;
  for(let i=1;i<axes.length;i++){
    const a=Math.atan2(axes[i-1][2],axes[i-1][1]),b=Math.atan2(axes[i][2],axes[i][1]);
    angle+=Math.atan2(Math.sin(b-a),Math.cos(b-a));
  }
  assert.ok(Math.abs(Math.abs(angle)-8*Math.PI)<1e-8,'body must rotate four complete turns about its longitudinal axis');
  for(const side of ['right','left']){
    strokes(5,8,15,`${side} foot`,2,false,5,25);
    strokes(5,16,23,`${side} foot`,2,false,6,20);
    strokes(5,16,23,`${side} foot`,1,true,3,55);
    strokes(5,16,23,`${side} foot`,1,false,3,55);
    strokes(5,24,29.4,`${side} foot`,2,false,3,15);
  }
});

test('Jump Splash Finale performs 3 closed jumps, 4 wide jumps, 6 side jumps and one high final leap',()=>{
  for(const side of ['right','left']){
    strokes(6,0,6,`${side} foot`,2,true,3,17);
    strokes(6,7,14,`${side} foot`,2,true,4,17);
    strokes(6,15,22,`${side} foot`,2,true,6,17);
    strokes(6,23,30,`${side} foot`,2,true,1,25);
  }
  const sideways=samples(6,15,22,p=>body(p,'pelvis').center[1]);
  assert.ok(Math.max(...sideways)-Math.min(...sideways)>20*ROBOT_GEOMETRY_SCALE,'side-jump phase must translate the whole body');
  const final=robotPose(30,1,6);
  assert.ok(Math.min(floor(body(final,'left knee')),floor(body(final,'right knee')))<3*ROBOT_GEOMETRY_SCALE,'final pose must kneel on one knee');
  assert.ok(Math.max(floor(body(final,'left knee')),floor(body(final,'right knee')))>8*ROBOT_GEOMETRY_SCALE,'final pose must not kneel on both knees');
});

test('floor routines keep supporting hands fixed and paired leg strokes retain constant separation',()=>{
  for(const [routine,start,endTime] of [[2,0,6],[2,7,14],[2,15,22],[5,8,15],[5,16,23]] as const){
    const initial=robotPose(start,1,routine);
    for(let frame=0;frame<=100;frame++){
      const pose=robotPose(start+(endTime-start)*frame/100,1,routine);
      for(const side of ['right','left']){
        const hand=body(pose,`${side} hand`);
        near(hand.center,body(initial,`${side} hand`).center,'planted hand moved during a floor-supported stroke');
        assert.ok(floor(hand)<3*ROBOT_GEOMETRY_SCALE,'supporting palm must be on the floor');
      }
      if(start===0)continue;
      const separation=sub(body(pose,'left foot').center,body(pose,'right foot').center);
      assert.ok(distance(separation,sub(body(initial,'left foot').center,body(initial,'right foot').center))<.001,
        'paired feet changed relative spacing');
      assert.ok(reach(pose,'left')>50.9*ROBOT_GEOMETRY_SCALE&&reach(pose,'right')>50.9*ROBOT_GEOMETRY_SCALE,'paired floor kicks must keep both legs extended');
    }
  }
});

test('all six moving boundaries use the velocity of their actual connected capsule endpoints',()=>{
  const scale=.375,dt=.00025,h=1e-5;
  for(const routine of ids)for(const t of [.41,2.63,5.43,9.27,14.41,18.31,22.51,26.73,29.71]){
    const previous=robotPose(t-h,1,routine),next=robotPose(t+h,1,routine),packed=packRobot(t,scale,dt,false,1,routine);
    for(let i=0;i<previous.length;i++)for(const sign of [-1,1]){
      const index=i*16,r:V3=[0,1,2].map(d=>sign*packed[index+12+d]*packed[index+7]) as V3;
      const omega:V3=[packed[index+8],packed[index+9],packed[index+10]];
      const rotation=[omega[1]*r[2]-omega[2]*r[1],omega[2]*r[0]-omega[0]*r[2],omega[0]*r[1]-omega[1]*r[0]];
      for(let d=0;d<3;d++){
        const derivative=(end(next[i],sign)[d]-end(previous[i],sign)[d])/(2*h)*scale*dt;
        assert.ok(Math.abs(packed[index+4+d]+rotation[d]-derivative)<3e-6,
          `routine ${routine} at ${t}s: ${previous[i].name} wall velocity does not match articulated geometry`);
      }
    }
  }
});

test('log-roll torso transfers all four turns to the tangential moving-wall velocity',()=>{
  const scale=.375,dt=.00025;
  for(const speed of [.5,1,4])for(const phaseTime of [.7,1.9,3.5,5.1,6.3]){
    const t=phaseTime/speed,h=1e-5/speed,pose=robotPose(t,speed,5);
    const torso=body(pose,'torso'),index=pose.indexOf(torso)*16;
    const packed=packRobot(t,scale,dt,false,speed,5);
    const omega:V3=[packed[index+8],packed[index+9],packed[index+10]];
    // Four axial revolutions over 0–7 s with a quintic start/stop envelope.
    // Endpoints alone cannot observe this spin because the torso's axis stays fixed.
    const u=phaseTime/7,expectedSpin=8*Math.PI/7*30*u*u*(1-u)*(1-u)*speed*dt;
    assert.ok(Math.abs(dot(omega,torso.axis)-expectedSpin)<1e-7,
      `log roll at ${phaseTime}s and ${speed}x omitted the torso's axial wall rotation`);
    const radial:V3=body(pose,'shoulders').axis.map(v=>v*torso.radius*scale) as V3;
    const rotation:V3=[omega[1]*radial[2]-omega[2]*radial[1],omega[2]*radial[0]-omega[0]*radial[2],omega[0]*radial[1]-omega[1]*radial[0]];
    // Track a material point on the torso skin using the independent shoulder bar.
    const skinPoint=(p:Capsule[]):V3=>{
      const trunk=body(p,'torso'),shoulders=body(p,'shoulders');
      return trunk.center.map((v,d)=>v+trunk.radius*shoulders.axis[d]) as V3;
    };
    const a=skinPoint(robotPose(t-h,speed,5)),b=skinPoint(robotPose(t+h,speed,5));
    for(let d=0;d<3;d++)assert.ok(Math.abs(packed[index+4+d]+rotation[d]-(b[d]-a[d])/(2*h)*scale*dt)<3e-6,
      `log roll at ${phaseTime}s: torso skin velocity differs from the rigid body's rotation`);
  }
});

