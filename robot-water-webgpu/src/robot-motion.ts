/** Prescribed articulated motion in reference-grid coordinates. +X is forward. */
export type V3=[number,number,number];
export interface Capsule {center:V3;axis:V3;halfLength:number;radius:number;material:number;name:string;}
const add=(a:V3,b:V3):V3=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const sub=(a:V3,b:V3):V3=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const mul=(a:V3,s:number):V3=>[a[0]*s,a[1]*s,a[2]*s];
const dot=(a:V3,b:V3)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a:V3,b:V3):V3=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=(v:V3):V3=>mul(v,1/Math.max(Math.hypot(...v),1e-12));
const clamp=(v:number,a=0,b=1)=>Math.max(a,Math.min(b,v));
const lerp=(a:number,b:number,u:number)=>a+(b-a)*u;
const mix=(a:V3,b:V3,u:number):V3=>[lerp(a[0],b[0],u),lerp(a[1],b[1],u),lerp(a[2],b[2],u)];
const smooth=(x:number)=>{const u=clamp(x);return u*u*u*(10+u*(-15+6*u));};
const TAU=2*Math.PI;
/** Half-size articulated robot inside the unchanged local pool. Geometry,
 * collision capsules and prescribed displacements use one uniform scale. */
export const ROBOT_GEOMETRY_SCALE=.5;
/** Reference dimensions map to exactly 60 × 60 × 70 cells at High (0.5625). */
export const ROBOT_DOMAIN=[320/3,320/3,1120/9] as const;
export const ROBOT_CENTER=[160/3,160/3] as const;
export const ROBOT_MAX_WATER_LEVEL=ROBOT_DOMAIN[2]*.7;
const AUTHORED_CENTER:V3=[128,96,0];
export const ROBOT_AUTHORED_SECONDS=30;
export const ROBOT_ROUTINE_SECONDS=10;
export const ROBOT_TIME_SCALE=ROBOT_AUTHORED_SECONDS/ROBOT_ROUTINE_SECONDS;
export const ROUTINES=[
  {id:1,name:'Water Slap Opening'}, {id:2,name:'Floor Sweep Storm'},
  {id:3,name:'Standing Kick Surge'}, {id:4,name:'Horse Stance Strikes'},
  {id:5,name:'Rolling Sweep Whirlwind'}, {id:6,name:'Jump Splash Finale'},
] as const;
export type RoutineId=1|2|3|4|5|6;
export interface DancePhase {start:number;end:number;repetitions:number;kind:string;}
/** Empty one-second intervals are deliberate transitions, not extra strokes. */
export const ROBOT_DANCE_PHASES:Record<RoutineId,readonly DancePhase[]>={
  1:[{start:0,end:5,repetitions:3,kind:'side slaps'},{start:6,end:12,repetitions:4,kind:'overhead forward slams'},
    {start:13,end:20,repetitions:5,kind:'squat outward scoops'},{start:21,end:30,repetitions:10,kind:'alternating side slaps'}],
  2:[{start:0,end:6,repetitions:4,kind:'prone alternating sweeps'},{start:7,end:14,repetitions:6,kind:'crab synchronous kicks'},
    {start:15,end:22,repetitions:4,kind:'crab paired side sweeps'},{start:23,end:30,repetitions:4,kind:'seated V slaps'}],
  3:[{start:0,end:7,repetitions:4,kind:'right forward kicks'},{start:8,end:14,repetitions:4,kind:'left forward kicks'},
    {start:15,end:22,repetitions:6,kind:'alternating outward kicks'},{start:23,end:30,repetitions:3,kind:'front to overhead scoops'}],
  4:[{start:0,end:8,repetitions:5,kind:'horse stance forward chops'},{start:9,end:16,repetitions:5,kind:'horse stance outward chops'},
    {start:17,end:24,repetitions:8,kind:'accelerating outward punches'},{start:25,end:30,repetitions:3,kind:'horse stance side slaps'}],
  5:[{start:0,end:7,repetitions:4,kind:'axial log rolls'},{start:8,end:15,repetitions:5,kind:'prone paired slaps'},
    {start:16,end:23,repetitions:6,kind:'supine paired pendulum slaps'},{start:24,end:30,repetitions:3,kind:'seated V slaps'}],
  6:[{start:0,end:6,repetitions:3,kind:'closed stance jumps'},{start:7,end:14,repetitions:4,kind:'open stance jumping slams'},
    {start:15,end:22,repetitions:6,kind:'alternating lateral jumps'},{start:23,end:30,repetitions:1,kind:'star jump and kneeling finish'}],
};

type Pair=[V3,V3];
interface Controls {
  root:V3; pitch:number; roll:number; yaw:number; nod:number;
  /** Limb targets are in the rigid pelvis frame; transitions never move bones separately. */
  hands:Pair; feet:Pair;
}
function rotate(p:V3,c:Controls,inverse=false):V3{
  const z=(v:V3,a:number):V3=>[Math.cos(a)*v[0]-Math.sin(a)*v[1],Math.sin(a)*v[0]+Math.cos(a)*v[1],v[2]];
  const y=(v:V3,a:number):V3=>[Math.cos(a)*v[0]+Math.sin(a)*v[2],v[1],-Math.sin(a)*v[0]+Math.cos(a)*v[2]];
  return inverse?z(y(z(p,-c.yaw),-c.pitch),-c.roll):z(y(z(p,c.roll),c.pitch),c.yaw);
}
const SIDES=[-1,1] as const;
const hip=(side:number):V3=>[0,7*side,0];
const shoulder=(side:number):V3=>[0,12*side,24];
function arm(c:Controls,i:number,direction:V3,reach=32){c.hands[i]=add(shoulder(SIDES[i]),mul(norm(direction),reach));}
function leg(c:Controls,i:number,direction:V3,reach=51){c.feet[i]=add(hip(SIDES[i]),mul(norm(direction),reach));}
function worldHand(c:Controls,i:number,target:V3){c.hands[i]=rotate(sub(target,c.root),c,true);}
function worldFoot(c:Controls,i:number,target:V3){c.feet[i]=rotate(sub(target,c.root),c,true);}
function base(rootZ=49,stance=23):Controls{
  const c:Controls={root:[128,96,rootZ],pitch:0,roll:0,yaw:0,nod:0,hands:[[0,0,0],[0,0,0]],feet:[[0,0,0],[0,0,0]]};
  for(const [i,side]of SIDES.entries()){
    worldFoot(c,i,[128,96+side*stance,6]);arm(c,i,[0,side,0]);
  }
  return c;
}
function grounded(c:Controls,stance=23){for(const[i,side]of SIDES.entries())worldFoot(c,i,[c.root[0],c.root[1]+side*stance,6]);}
function sideArms(c:Controls,angles:number[]){for(const[i,side]of SIDES.entries())arm(c,i,[-.35,side*Math.cos(angles[i]),Math.sin(angles[i])]);}
function frontArms(c:Controls,angle:number){for(let i=0;i<2;i++)arm(c,i,[Math.cos(angle),0,Math.sin(angle)]);}
/** A complete stroke goes home -> contact -> home, C2 at every turnaround. */
function beat(t:number,start:number,end:number,count:number,accelerating=false){
  const progress=clamp((t-start)/(end-start)),travel=accelerating?.55*progress+.45*progress*progress:progress;
  const cycle=Math.min(count-1,Math.floor(travel*count)),fraction=travel===1?1:travel*count-cycle;
  const pulse=fraction<=.5?smooth(2*fraction):1-smooth(2*fraction-1);
  return {pulse,fraction,index:cycle,progress};
}
function curve(x:number,keys:[number,number][]):number{
  for(let i=1;i<keys.length;i++)if(x<keys[i][0])return lerp(keys[i-1][1],keys[i][1],smooth((x-keys[i-1][0])/(keys[i][0]-keys[i-1][0])));
  return keys[keys.length-1][1];
}
function finalRaise(c:Controls,t:number,start:number,end:number,overhead=false){
  const u=smooth((t-start)/(end-start));
  for(const[i,side]of SIDES.entries())c.hands[i]=mix(c.hands[i],add(shoulder(side),mul(norm(overhead?[0,0,1]:[0,side,.85]),32)),u);
}
/** Real knee flexion supplies the reach that an upright knee-deep robot lacks. */
function sideSlap(t:number,p: DancePhase,water:number,horse=false,alternating=false):Controls{
  const b=beat(t,p.start,p.end-(p.end===30?.65:0),p.repetitions),amount=b.pulse;
  const high=horse?Math.min(30,water+3):49,low=Math.max(13,Math.min(high,water+4));
  const c=base(lerp(high,low,amount),horse?29:24);
  const down=Math.asin(clamp((water-3-(low+24))/32,-.99,.99));
  const angles=[0,0];
  if(alternating){angles[b.index%2]=lerp(0,down,amount);angles[1-b.index%2]=.3*amount;}
  else angles.fill(lerp(0,down,amount));
  sideArms(c,angles);
  if(p.end===30)finalRaise(c,t,29.35,30);
  return c;
}
function forwardSlams(t:number,p:DancePhase,water:number,horse=false):Controls{
  const b=beat(t,p.start,p.end,p.repetitions),high=horse?Math.min(30,water+3):49;
  const low=Math.max(11,Math.min(high,water+1)),c=base(lerp(high,low,b.pulse),horse?29:24);
  const strike=Math.max(-1.1,Math.asin(clamp((water-3-low-24)/32,-.99,.99)));
  frontArms(c,lerp(Math.PI/2,strike,b.pulse));return c;
}
function outwardScoops(t:number,p:DancePhase,water:number):Controls{
  const b=beat(t,p.start,p.end,p.repetitions),c=base(Math.max(14,Math.min(30,water-4)),29);
  const submerged=Math.asin(clamp((water-7-c.root[2]-24)/32,-.99,.99));
  const angle=curve(b.fraction,[[0,0],[.28,submerged],[.69,.7],[1,0]]);
  sideArms(c,[angle,angle]);return c;
}
function floorBase(supine:boolean,rootZ=24):Controls{
  const c=base(rootZ,14);c.pitch=-Math.PI/2;c.roll=supine?0:Math.PI;
  // Each wrist is planted underneath its own shoulder, outside the chassis.
  for(const[i,side]of SIDES.entries()){
    const s=add(c.root,rotate(shoulder(side),c));
    worldHand(c,i,[s[0]-2,s[1]+(supine?side:-side)*4,4]);
  }
  return c;
}
function extendedWorldLeg(c:Controls,i:number,forward:number,lateral:number,rise:number){
  const h=add(c.root,rotate(hip(SIDES[i]),c));
  const direction=norm([forward,lateral,rise]);worldFoot(c,i,add(h,mul(direction,51)));
}
/** Keep a paired ankle track distinct from the shoulder-width track without stretching hips. */
function pairedLegs(c:Controls,separation:number){
  const direction=norm(mul(add(c.feet[0],c.feet[1]),.5)),offset=separation*.5-7;
  const lateral=Math.abs(direction[1]*offset);
  const reach=Math.sqrt(51*51-offset*offset+lateral*lateral)-lateral-.025;
  const center=mul(direction,reach);
  for(const[i,side]of SIDES.entries())c.feet[i]=add(center,[0,side*separation*.5,0]);
}
function proneSweeps(t:number,p:DancePhase,water:number):Controls{
  const c=floorBase(false),b=beat(t,p.start,p.end,p.repetitions);
  for(const[i,side]of SIDES.entries()){
    const a=i===b.index%2?1.05*b.pulse:0;
    // Local right is world +Y in this face-down orientation.
    const rise=(water+2-c.root[2])/51;
    extendedWorldLeg(c,i,Math.cos(a),-side*Math.sin(a),rise);
  }
  return c;
}
function crabKicks(t:number,p:DancePhase,water:number):Controls{
  const c=floorBase(true),b=beat(t,p.start,p.end,p.repetitions);
  const low=Math.asin(clamp((water-7-c.root[2])/51,-.8,.8)),angle=lerp(.78,low,b.pulse);
  for(let i=0;i<2;i++)extendedWorldLeg(c,i,Math.cos(angle),0,Math.sin(angle));
  pairedLegs(c,24);
  return c;
}
function crabSweeps(t:number,p:DancePhase,water:number):Controls{
  const c=floorBase(true),u=clamp((t-p.start)/(p.end-p.start));
  const yaw=.83*Math.sin(TAU*p.repetitions*smooth(u)),rise=(water-1-c.root[2])/51;
  for(let i=0;i<2;i++)extendedWorldLeg(c,i,Math.cos(yaw),Math.sin(yaw),rise);
  pairedLegs(c,14);
  return c;
}
function seatedV(t:number,p:DancePhase,water:number):Controls{
  const b=beat(t,p.start,p.end-.6,p.repetitions),c=base(10,20);c.pitch=-.12;
  const low=Math.asin(clamp((water-6-c.root[2])/51,-.8,.8)),high=Math.min(.96,low+.45);
  const angle=lerp(high,low,b.pulse);
  for(const[i,side]of SIDES.entries())extendedWorldLeg(c,i,.76*Math.cos(angle),side*.65*Math.cos(angle),Math.sin(angle));
  sideArms(c,[0,0]);finalRaise(c,t,p.end-.6,p.end);return c;
}
function forwardKicks(t:number,p:DancePhase,water:number,kicking:number):Controls{
  const c=base(56.3,12),b=beat(t,p.start,p.end,p.repetitions),angle=lerp(-.23,1.15,b.pulse);
  // Supporting foot remains on the floor, while the other leg swings in its own sagittal plane.
  leg(c,kicking,[Math.sin(angle),0,-Math.cos(angle)]);
  return c;
}
function outwardKicks(t:number,p:DancePhase):Controls{
  const c=base(56.3,15),b=beat(t,p.start,p.end,p.repetitions),k=b.index%2,side=SIDES[k];
  const angle=lerp(.13,1.35,b.pulse);leg(c,k,[0,side*Math.sin(angle),-Math.cos(angle)]);
  return c;
}
function overheadScoops(t:number,p:DancePhase,water:number):Controls{
  const b=beat(t,p.start,p.end-.7,p.repetitions),high=49,low=Math.max(13,Math.min(31,water+1));
  const plunge=curve(b.fraction,[[0,0],[.22,1],[.7,0],[1,0]]),c=base(lerp(high,low,plunge),26);
  const angle=curve(b.fraction,[[0,.05],[.22,-1.18],[.7,1.78],[1,.05]]);
  frontArms(c,angle);finalRaise(c,t,p.end-.7,p.end,true);return c;
}
function outwardChops(t:number,p:DancePhase,water:number):Controls{
  const c=base(Math.max(14,Math.min(30,water+1)),29),b=beat(t,p.start,p.end,p.repetitions);
  const low=Math.asin(clamp((water-5-c.root[2]-24)/32,-.99,.99));
  sideArms(c,[lerp(Math.PI/2,low,b.pulse),lerp(Math.PI/2,low,b.pulse)]);return c;
}
function outwardPunches(t:number,p:DancePhase,water:number):Controls{
  // Below knee-deep water, preserve a reachable horse stance rather than folding
  // elbows through the thighs in an attempt to reach an arbitrarily low surface.
  const b=beat(t,p.start,p.end,p.repetitions,true),c=base(Math.max(23,Math.min(27,water-7)),29);
  const vertical=clamp(water-3-c.root[2]-24,-28,-8);
  const forward=lerp(-6,12,smooth((24-water)/12));
  for(const[i,side]of SIDES.entries()){
    const home:V3=[12,side*16,17];
    const target=add(shoulder(side),[forward,side*Math.sqrt(32*32-forward*forward-vertical*vertical),vertical]);
    c.hands[i]=mix(home,target,i===b.index%2?b.pulse:0);
  }
  return c;
}
function logRoll(t:number,p:DancePhase):Controls{
  const c=base(13,7);c.pitch=-Math.PI/2;c.roll=-Math.PI/2+TAU*p.repetitions*smooth((t-p.start)/(p.end-p.start));
  c.root[1]+=9*Math.sin(TAU*smooth((t-p.start)/(p.end-p.start)));
  arm(c,0,[0,0,1]);arm(c,1,[0,.17,-1]);
  for(let i=0;i<2;i++)leg(c,i,[0,0,-1]);pairedLegs(c,14);return c;
}
function proneSlaps(t:number,p:DancePhase,water:number):Controls{
  const c=floorBase(false),b=beat(t,p.start,p.end,p.repetitions);
  const low=Math.asin(clamp((water-7-c.root[2])/51,-.8,.8)),angle=lerp(.7,low,b.pulse);
  for(let i=0;i<2;i++)extendedWorldLeg(c,i,Math.cos(angle),0,Math.sin(angle));pairedLegs(c,14);return c;
}
function supinePendulum(t:number,p:DancePhase,water:number):Controls{
  const c=floorBase(true,12),b=beat(t,p.start,p.end,p.repetitions);
  // Arms form a rigid T on the floor; a single common rotation moves the leg pair.
  for(const[i,side]of SIDES.entries())worldHand(c,i,[104,96+side*43,4]);
  const roll=(b.index%2===0?-1:1)*1.22*b.pulse;
  const a=.78,vertical=Math.sin(a)*Math.cos(roll),lateral=Math.sin(a)*Math.sin(roll);
  for(let i=0;i<2;i++)extendedWorldLeg(c,i,Math.cos(a),lateral,vertical);pairedLegs(c,14);return c;
}
function jumping(t:number,p:DancePhase,water:number,mode:0|1|2):Controls{
  const b=beat(t,p.start,p.end,p.repetitions),f=b.fraction;
  const crouch=curve(f,[[0,0],[.18,1],[.35,0],[.64,0],[.79,1],[1,0]]);
  const airborne=curve(f,[[0,0],[.25,0],[.46,1],[.7,0],[1,0]]),strike=curve(f,[[0,0],[.62,0],[.77,1],[1,0]]);
  const stance=mode===0?9:mode===1?27:22;
  const low=Math.max(15,Math.min(34,water+3)),c=base(49-(49-low)*crouch,stance);
  c.root[2]+=19*airborne;
  if(mode===2){
    const side=b.index%2===0?1:-1,progress=smooth(clamp((f-.23)/.5));
    c.root[1]=96+(b.index===0?0:-side*12)+side*(b.index===0?12:24)*progress;
  }
  const footHeight=6+19*airborne;
  for(const[i,side]of SIDES.entries())worldFoot(c,i,[c.root[0],c.root[1]+side*stance,footHeight]);
  const down=Math.max(-1.1,Math.asin(clamp((water-4-c.root[2]-24)/32,-.99,.99)));
  if(mode===1)frontArms(c,lerp(Math.PI/2,down,strike));
  else{
    const angles=[0,0];if(mode===2)angles[b.index%2]=down*strike;else angles.fill(down*strike);
    sideArms(c,angles);
  }
  return c;
}
function starFinale(t:number,p:DancePhase,water:number):Controls{
  const f=(t-p.start)/(p.end-p.start),crouch=curve(f,[[0,0],[.21,1],[.4,0],[.56,0],[.72,1],[.83,1],[1,1]]);
  const jump=curve(f,[[0,0],[.24,0],[.43,1],[.64,0],[1,0]]),spread=curve(f,[[0,0],[.25,0],[.43,1],[.58,0],[1,0]]);
  const c=base(49-23*crouch+29*jump,9);
  for(const[i,side]of SIDES.entries())worldFoot(c,i,[128,96+side*(9+25*spread),6+29*jump]);
  const armAngle=curve(f,[[0,-1.15],[.2,-1.25],[.42,0],[.58,-1],[.68,-1.25],[.85,.7],[1,.7]]);
  sideArms(c,[armAngle,armAngle]);
  const windup=curve(f,[[0,0],[.2,1],[.38,0],[1,0]]);
  for(const[i,side]of SIDES.entries())c.hands[i]=mix(c.hands[i],add(shoulder(side),mul(norm([-.8,side*.12,-.6]),32)),windup);
  const kneel=smooth((f-.76)/.2);
  c.feet[0]=mix(c.feet[0],[26,-11,6-c.root[2]],kneel);
  c.feet[1]=mix(c.feet[1],[-22,11,7-c.root[2]],kneel);
  c.nod=-.1*kneel;return c;
}
function phaseControls(routine:RoutineId,index:number,t:number,water:number):Controls{
  const p=ROBOT_DANCE_PHASES[routine][index];
  if(routine===1)return index===0?sideSlap(t,p,water):index===1?forwardSlams(t,p,water):index===2?outwardScoops(t,p,water):sideSlap(t,p,water,false,true);
  if(routine===2)return index===0?proneSweeps(t,p,water):index===1?crabKicks(t,p,water):index===2?crabSweeps(t,p,water):seatedV(t,p,water);
  if(routine===3)return index===0?forwardKicks(t,p,water,0):index===1?forwardKicks(t,p,water,1):index===2?outwardKicks(t,p):overheadScoops(t,p,water);
  if(routine===4)return index===0?forwardSlams(t,p,water,true):index===1?outwardChops(t,p,water):index===2?outwardPunches(t,p,water):sideSlap(t,p,water,true);
  if(routine===5)return index===0?logRoll(t,p):index===1?proneSlaps(t,p,water):index===2?supinePendulum(t,p,water):seatedV(t,p,water);
  return index===3?starFinale(t,p,water):jumping(t,p,water,index as 0|1|2);
}
function interpolate(a:Controls,b:Controls,u:number):Controls{
  const angle=(x:number,y:number)=>x+Math.atan2(Math.sin(y-x),Math.cos(y-x))*u;
  return {root:mix(a.root,b.root,u),pitch:angle(a.pitch,b.pitch),roll:angle(a.roll,b.roll),yaw:angle(a.yaw,b.yaw),nod:lerp(a.nod,b.nod,u),
    hands:[mix(a.hands[0],b.hands[0],u),mix(a.hands[1],b.hands[1],u)],feet:[mix(a.feet[0],b.feet[0],u),mix(a.feet[1],b.feet[1],u)]};
}
function controls(routine:RoutineId,t:number,water:number):Controls{
  const phases=ROBOT_DANCE_PHASES[routine];
  for(let i=0;i<phases.length;i++){
    const p=phases[i];
    if(t<=p.end)return phaseControls(routine,i,Math.max(t,p.start),water);
    const next=phases[i+1];
    if(next&&t<next.start)return interpolate(phaseControls(routine,i,p.end,water),phaseControls(routine,i+1,next.start,water),smooth((t-p.end)/(next.start-p.end)));
  }
  return phaseControls(routine,3,30,water);
}
/** Solve a hinge chain in the rigid body frame, then transform the complete skeleton once. */
function joint(origin:V3,target:V3,a:number,b:number,pole:V3):[V3,V3]{
  const delta=sub(target,origin),axis=Math.hypot(...delta)>1e-10?norm(delta):[0,0,-1] as V3;
  // A microscopic smooth extension margin removes the IK singularity at a
  // perfectly straight hinge, while leaving bone lengths exactly unchanged.
  const requested=Math.hypot(...delta),gap=a+b-requested;
  const distance=Math.max(Math.abs(a-b)+.000001,a+b-.5*(gap+Math.sqrt(gap*gap+.0001)));
  const along=(a*a-b*b+distance*distance)/(2*distance);
  let perpendicular=sub(pole,mul(axis,dot(axis,pole)));
  if(Math.hypot(...perpendicular)<1e-8)perpendicular=sub([1,0,0],mul(axis,axis[0]));
  if(Math.hypot(...perpendicular)<1e-8)perpendicular=sub([0,1,0],mul(axis,axis[1]));
  return [add(add(origin,mul(axis,along)),mul(norm(perpendicular),Math.sqrt(Math.max(0,a*a-along*along)))),add(origin,mul(axis,distance))];
}

/** Six separate 30-second scores. Water level affects reach, never limb length. */
export function robotPose(seconds:number,speed=1,routine:RoutineId=1,waterLevel=15):Capsule[]{
  if(!Number.isFinite(seconds)||!Number.isFinite(speed)||speed<=0)throw new Error('Invalid robot choreography time or speed.');
  if(!Number.isInteger(routine)||routine<1||routine>6)throw new Error('Routine must be an integer from 1 to 6.');
  if(!Number.isFinite(waterLevel)||waterLevel<4||waterLevel>ROBOT_MAX_WATER_LEVEL)throw new Error(`Water level must be between 4 and ${ROBOT_MAX_WATER_LEVEL.toFixed(3)}.`);
  // At deep settings the fluid rises independently; IK reach stays clamped to
  // the highest authored waterline so bones never stretch toward the surface.
  const authoredWaterLevel=clamp(waterLevel/ROBOT_GEOMETRY_SCALE,8,50);
  const t=clamp(seconds*speed,0,ROBOT_AUTHORED_SECONDS),c=controls(routine,t,authoredWaterLevel),result:Capsule[]=[];
  // During routine 5's supine-to-seated transition, a smooth out-of-plane IK
  // pole bias prevents the leg target and bend pole from becoming parallel.
  // It is exactly zero, with zero slope, at both authored endpoint poses.
  const transitionU=clamp(t-23),legPoleBias=routine===5?.45*16*transitionU*transitionU*(1-transitionU)*(1-transitionU):0;
  const world=(p:V3)=>add(c.root,rotate(p,c));
  const bone=(name:string,a:V3,b:V3,radius:number,material=0)=>{
    const d=sub(b,a),length=Math.hypot(...d);
    result.push({name,center:mul(add(a,b),.5),axis:length>1e-10?mul(d,1/length):rotate([0,0,1],c),halfLength:length*.5,radius,material});
  };
  bone('pelvis',world([0,-7,0]),world([0,7,0]),3.8,1);
  bone('torso',world([0,0,8]),world([0,0,22]),4.3);
  bone('shoulders',world([0,-12,24]),world([0,12,24]),2.7,1);
  bone('neck',world([0,0,27]),world([0,0,31]),2.4,1);
  const head=(p:V3)=>world([p[0]*Math.cos(c.nod)+(p[2]-31)*Math.sin(c.nod),p[1],31-p[0]*Math.sin(c.nod)+(p[2]-31)*Math.cos(c.nod)]);
  bone('head',head([0,0,34]),head([0,0,39]),4.5);
  bone('visor',head([4.5,-2.8,37]),head([4.5,2.8,37]),1.8,2);
  for(const[i,side]of SIDES.entries()){
    const name=side<0?'right':'left',h=hip(side),s=shoulder(side);
    const lateralReach=Math.abs(c.feet[i][1]-h[1]),wide=smooth((lateralReach-8)/12),lowWater=smooth((24-authoredWaterLevel)/12);
    const outward=smooth((lateralReach-6)/9);
    const [knee,ankle]=joint(h,c.feet[i],25.5,25.5,[lerp(lerp(1,.35,wide),lerp(1,.04,outward),lowWater),side*lerp(.3,1,wide),legPoleBias]);
    bone(`${name} thigh`,world(h),world(knee),3);
    bone(`${name} shin`,world(knee),world(ankle),2.5);
    bone(`${name} knee`,world(knee),world(knee),3.2,1);
    // The ankle is inside the heel envelope, so feet never detach on a roll.
    const foot=add(ankle,[1.5,0,-1.3]);
    bone(`${name} foot`,world(add(foot,[-5,0,0])),world(add(foot,[5,0,0])),2.5,1);
    // Low horse-stance strokes route the elbows forward of the bent knees.
    const [elbow,hand]=joint(s,c.hands[i],17,15,[1,side*lerp(1.4,.3,routine===4?lowWater:0),.2]);
    bone(`${name} upper arm`,world(s),world(elbow),2.2);
    bone(`${name} forearm`,world(elbow),world(hand),2);
    bone(`${name} elbow`,world(elbow),world(elbow),2.5,1);
    bone(`${name} hand`,world(hand),world(hand),2.4,1);
  }
  // A single rigid support correction keeps every articulated endpoint attached.
  // Smooth minimum/positive-part avoid impulses when floor contact changes body part.
  const bottoms=result.map(b=>b.center[2]-Math.sqrt((b.axis[2]*b.halfLength)**2+.02**2)-b.radius);
  const minimum=Math.min(...bottoms),softMin=minimum-.035*Math.log(bottoms.reduce((sum,z)=>sum+Math.exp((minimum-z)/.035),0));
  const penetration=1.4-softMin,lift=Math.max(0,penetration)+.025*Math.log1p(Math.exp(-Math.abs(penetration)/.025));
  for(const b of result){
    const center=add(b.center,[0,0,lift]);
    b.center=[ROBOT_CENTER[0]+(center[0]-AUTHORED_CENTER[0])*ROBOT_GEOMETRY_SCALE,
      ROBOT_CENTER[1]+(center[1]-AUTHORED_CENTER[1])*ROBOT_GEOMETRY_SCALE,
      center[2]*ROBOT_GEOMETRY_SCALE];
    b.halfLength*=ROBOT_GEOMETRY_SCALE;b.radius*=ROBOT_GEOMETRY_SCALE;
  }
  return result;
}

/** 16 floats: center/radius, linear velocity/half-length, omega/material, axis/pad. */
export function packRobot(seconds:number,scale:number,secondsPerStep:number,stationary=false,speed=1,routine:RoutineId=1,waterLevel=15):Float32Array{
  if(!Number.isFinite(seconds)||seconds<0||!(scale>0)||!(secondsPerStep>0))throw new Error('Invalid robot time or grid mapping.');
  const time=stationary?0:seconds,pose=robotPose(time,speed,routine,waterLevel),h=.0001/speed;
  const stopped=stationary||seconds*speed>=ROBOT_AUTHORED_SECONDS;
  const before=robotPose(Math.max(0,time-h),speed,routine,waterLevel),after=robotPose(time+h,speed,routine,waterLevel),delta=time<h?time+h:2*h;
  // A capsule axis supplies only the two angular components perpendicular to
  // itself. Recover the chassis rotation from its full orthonormal frame so a
  // longitudinal log roll also moves the torso's wall tangentially. Limb hinges
  // supply the remaining transverse rotation; axial twist follows the chassis.
  const frame=(bodies:Capsule[]):[V3,V3,V3]=>{
    const y=bodies[0].axis,z=bodies[1].axis;
    return [cross(y,z),y,z];
  };
  const basis=frame(pose),basisBefore=frame(before),basisAfter=frame(after);
  let chassisOmega:V3=[0,0,0];
  if(!stopped)for(let j=0;j<3;j++)chassisOmega=add(chassisOmega,mul(cross(basis[j],sub(basisAfter[j],basisBefore[j])),.5*secondsPerStep/delta));
  const data=new Float32Array(pose.length*16);
  pose.forEach((body,i)=>{
    const v=stopped?[0,0,0] as V3:mul(sub(after[i].center,before[i].center),scale*secondsPerStep/delta);
    const transverse=mul(cross(body.axis,sub(after[i].axis,before[i].axis)),secondsPerStep/delta);
    const omega=stopped?[0,0,0] as V3:add(transverse,mul(body.axis,dot(chassisOmega,body.axis)));
    data.set([...mul(body.center,scale),body.radius*scale,...v,body.halfLength*scale,...omega,body.material,...body.axis,0],i*16);
  });return data;
}
export function capsuleSignedDistance(point:V3,body:Capsule):number{
  const r=sub(point,body.center),along=clamp(dot(r,body.axis),-body.halfLength,body.halfLength);
  return Math.hypot(...sub(r,mul(body.axis,along)))-body.radius;
}

