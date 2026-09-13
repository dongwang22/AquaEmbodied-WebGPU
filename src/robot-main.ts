import './robot-style.css';
import {FluidGPU} from './gpu';
import {View3D} from './view3d';
import {packRobot,ROBOT_MAX_WATER_LEVEL,ROBOT_ROUTINE_SECONDS,ROBOT_TIME_SCALE,ROUTINES,type RoutineId} from './robot-motion';
import {routineStageLabel} from './robot-routine-info';
import {readRobotControlPresets,readRobotParameters,ROBOT_PRESET_NAMES,robotFluidConfig,robotTimeStep,type RobotParameters,type RobotPresetGroup} from './robot-config';

document.querySelector<HTMLDivElement>('#app')!.innerHTML=`
  <header><div class="brand"><span class="brand-mark">R</span><div><h1>Robot Dance</h1><p>WebGPU · LBM–VOF</p></div></div></header>
  <section class="viewport"><canvas id="scene" tabindex="0" aria-label="3D robot dancing in knee-deep water"></canvas>
    <div class="scene-label"><span class="tag" id="routine-name">10-SECOND CHOREOGRAPHY</span><p id="phase">Preparing the routine…</p></div>
    <div class="clock"><output id="time">0.000 s</output><span>Simulation time</span></div>
    <aside class="panel"><h2>Dance in water</h2><p class="intro">Choose one of six 10-second routines. Every pose uses the same connected articulated skeleton.</p>
      <label for="resolution">Resolution</label><select id="resolution"><option value="0.375">40 × 40 × 47 · Fast</option><option value="0.4375">47 × 47 × 54 · Refined</option><option value="0.5625" selected>60 × 60 × 70 · High</option></select>
      <label for="nu">Viscosity</label><select id="nu"><option value="0.0008">Small · 0.0008</option><option value="0.004" selected>Medium · 0.004</option><option value="0.02">Large · 0.02</option></select>
      <label for="sigma">Surface tension</label><select id="sigma"><option value="0.0002">Small · 0.0002</option><option value="0.001" selected>Medium · 0.001</option><option value="0.005">Large · 0.005</option></select>
      <label for="water-level">Water level (lattice)</label><input id="water-level" type="number" min="4" max="${ROBOT_MAX_WATER_LEVEL}" step="0.1" value="15">
      <label for="routine">Dance routine</label><select id="routine">${ROUTINES.map(r=>`<option value="${r.id}">${r.id}. ${r.name}</option>`).join('')}</select>
      <label for="motion">Motion speed (×)</label><input id="motion" type="number" min="0.5" max="4" step="0.1" value="1">
      <label for="duration">Run time (s)</label><input id="duration" type="number" min="0.01" max="120" step="0.1" value="${ROBOT_ROUTINE_SECONDS}">
      <div class="actions"><button id="play" disabled>Start dance</button><button id="reset" disabled>Apply &amp; reset</button></div>
      <p id="status" role="status" aria-live="polite">Preparing the simulation…</p>
      <p class="note">1× completes the proportional routine in 10 s. High uses a 60 × 60 × 70 domain centered on the robot. Speed changes apply on reset. The final pose is held after the routine ends. The four side boundaries are periodic; the floor and ceiling are solid walls. Live playback depends on your GPU.</p>
    </aside>
    <p class="controls">Drag to rotate · Scroll to zoom · Right drag to pan · Double-click to reset view</p>
  </section>`;
const canvas=document.querySelector<HTMLCanvasElement>('#scene')!;
const play=document.querySelector<HTMLButtonElement>('#play')!,reset=document.querySelector<HTMLButtonElement>('#reset')!;
const resolution=document.querySelector<HTMLSelectElement>('#resolution')!,duration=document.querySelector<HTMLInputElement>('#duration')!;
const status=document.querySelector<HTMLParagraphElement>('#status')!,clock=document.querySelector<HTMLOutputElement>('#time')!;
let params:RobotParameters,fluid:FluidGPU|undefined,view:View3D|undefined;
let running=false,step=0,credit=0,last=0,initializing=false,work:Promise<void>|undefined;
let stationaryPose=false;
function loadPresetOptions(select:HTMLSelectElement,group:RobotPresetGroup,useJsonDefault:boolean){
  const previous=Math.max(0,select.selectedIndex),selected=useJsonDefault?ROBOT_PRESET_NAMES.indexOf(group.default):Math.min(previous,ROBOT_PRESET_NAMES.length-1);
  select.innerHTML=ROBOT_PRESET_NAMES.map((name,index)=>`<option value="${group[name]}"${index===selected?' selected':''}>${name[0].toUpperCase()+name.slice(1)} · ${group[name]}</option>`).join('');
}
const time=()=>params?step*robotTimeStep(params):0;
const sceneBodies=(t:number,stationary=stationaryPose)=>packRobot(t,params.gridScale,robotTimeStep(params),stationary,params.motionSpeed*ROBOT_TIME_SCALE,params.routine as RoutineId,params.waterLevel);
const fail=(error:unknown)=>{running=false;play.textContent='Start dance';play.disabled=true;status.textContent=error instanceof Error?error.message:String(error);status.classList.add('error');};
function updateView(){if(fluid&&params){view?.updateRobot(sceneBodies(time()));clock.textContent=`${time().toFixed(3)} s`;
    const t=time()*params.motionSpeed*ROBOT_TIME_SCALE,routine=ROUTINES.find(r=>r.id===params.routine)!;
    document.querySelector('#routine-name')!.textContent=`${routine.id}. ${routine.name}`;
    document.querySelector('#phase')!.textContent=routineStageLabel(params.routine as RoutineId,t);
}}
async function advance(count:number,stationary=false,frameBudget=Infinity){
  if(!fluid)throw new Error('Simulation is not ready.');
    stationaryPose=stationary;
    const started=performance.now();let completed=0;
    const dt=robotTimeStep(params);
    for(let remaining=count;remaining>0;){
    const size=Math.min(remaining,params.stepsPerBatch),poses=Array.from({length:size},(_,k)=>sceneBodies((step+k+1)*dt));
    await fluid.advanceKinematic(poses);
    step+=size;remaining-=size;completed+=size;
    if(performance.now()-started>=frameBudget)break;
  }
  updateView();
  return completed;
}
async function initialize(useJsonResolution=false){
  if(initializing)return;initializing=true;running=false;play.disabled=true;reset.disabled=true;resolution.disabled=true;
  status.classList.remove('error');status.textContent='Preparing the simulation…';
  try{
    await work;view?.destroy();view=undefined;fluid?.destroy();fluid=undefined;
    const response=await fetch(`${import.meta.env.BASE_URL}robotInputParameter.json`,{cache:'no-store'});
    if(!response.ok)throw new Error(`Cannot load robotInputParameter.json (${response.status}).`);
    const raw=await response.json(),presets=readRobotControlPresets(raw);
    loadPresetOptions(document.querySelector<HTMLSelectElement>('#nu')!,presets.viscosity,useJsonResolution);
    loadPresetOptions(document.querySelector<HTMLSelectElement>('#sigma')!,presets.surfaceTension,useJsonResolution);
    params=readRobotParameters(raw);
    if(useJsonResolution){resolution.value=String(params.gridScale);duration.value=String(params.durationSeconds);(document.querySelector('#motion') as HTMLInputElement).value=String(params.motionSpeed);(document.querySelector('#water-level') as HTMLInputElement).value=String(params.waterLevel);(document.querySelector('#routine') as HTMLSelectElement).value=String(params.routine);}
    else params=readRobotParameters({...params,gridScale:Number(resolution.value),durationSeconds:Number(duration.value)});
    params=readRobotParameters({...params,viscosity:Number((document.querySelector('#nu') as HTMLSelectElement).value),surfaceTension:Number((document.querySelector('#sigma') as HTMLSelectElement).value),waterLevel:Number((document.querySelector('#water-level') as HTMLInputElement).value),motionSpeed:Number((document.querySelector('#motion') as HTMLInputElement).value),routine:Number((document.querySelector('#routine') as HTMLSelectElement).value)});
    step=0;credit=0;stationaryPose=false;
    const dt=robotTimeStep(params);
    fluid=await FluidGPU.create(robotFluidConfig(params),sceneBodies(0),{precision:'fp32'});
    view=await View3D.create(canvas,fluid,fail,{waterOpacity:.65,targetFps:60,pixelBudget:250000,cameraYaw:1.05,cameraPitch:.36,cameraDistance:170,cameraCenter:[160/3,160/3,26]});
    updateView();status.textContent='Ready';play.disabled=false;play.textContent='Start dance';
  }catch(error){fail(error);}finally{initializing=false;reset.disabled=false;resolution.disabled=false;}
}
play.addEventListener('click',()=>{
  if(!fluid||initializing)return;
  const seconds=Number(duration.value);if(!Number.isFinite(seconds)||seconds<=0||seconds>120){status.textContent='Enter a run time greater than 0 and no more than 120 s.';return;}
  params.durationSeconds=seconds;
  if(!running&&time()>=seconds){status.textContent='Reset to start a new run, or increase the run time.';return;}
  running=!running;last=performance.now();credit=0;
  play.textContent=running?'Pause':'Continue';status.textContent=running?'Dancing':'Paused';
});
reset.addEventListener('click',()=>void initialize());
resolution.addEventListener('change',()=>void initialize());
document.querySelector('#routine')!.addEventListener('change',()=>{
  running=false;play.disabled=true;play.textContent='Start dance';
  status.textContent='Click Apply & reset to load this routine.';
});
const tick=(now:number)=>{
  requestAnimationFrame(tick);
  if(!running||initializing||!fluid){last=now;return;}
  credit=Math.min(.1,credit+Math.max(0,now-last)/1000);last=now;
  if(work)return;
  const dt=robotTimeStep(params),limit=Math.floor(params.durationSeconds/dt+1e-7);
  // A render frame can contain several ordered LBM substeps; batched command
  // encoding keeps the pure prescribed robot scene responsive.
  const count=Math.min(128,limit-step,Math.floor(credit/dt));
  if(count<=0){if(step>=limit){running=false;play.textContent='Start dance';status.textContent='Run complete';}return;}
  work=advance(count,false,12).then(done=>{credit=Math.max(0,credit-done*dt);}).catch(fail).finally(()=>{work=undefined;});
};
requestAnimationFrame(tick);
// Local validation only; production builds do not expose simulation internals.
if(import.meta.env.DEV)Object.assign(window,{robotDemo:{
  get fluid(){return fluid;},get parameters(){return {...params};},get time(){return time();},get running(){return running;},get camera(){return view?.camera;},
  reset:async(scale?:number)=>{if(scale!==undefined)resolution.value=String(scale);await initialize();},
  advance:async(count:number,stationary=false)=>{running=false;await work;await advance(count,stationary);},
}});
await initialize(true);
