import type {FluidConfig} from './gpu';
import {ROBOT_DOMAIN,ROBOT_GEOMETRY_SCALE,ROBOT_MAX_WATER_LEVEL,ROBOT_TIME_SCALE} from './robot-motion';
export interface RobotParameters{
  gridScale:number;viscosity:number;surfaceTension:number;gravity:number;smagorinsky:number;
  waterLevel:number;robotContactAngle:number;wallContactAngle:number;
  secondsPerStep:number;durationSeconds:number;stepsPerBatch:number;
  motionSpeed:number;
  timeRefinement:number;
  routine:number;
}
export const ROBOT_PRESET_NAMES=['small','medium','large'] as const;
export type RobotPresetName=typeof ROBOT_PRESET_NAMES[number];
export interface RobotPresetGroup{small:number;medium:number;large:number;default:RobotPresetName;}
export interface RobotControlPresets{viscosity:RobotPresetGroup;surfaceTension:RobotPresetGroup;}
const FALLBACK_PRESETS:RobotControlPresets={
  viscosity:{small:.0008,medium:.004,large:.02,default:'medium'},
  surfaceTension:{small:.0002,medium:.001,large:.005,default:'medium'},
};
function presetGroup(value:unknown,fallback:RobotPresetGroup,allowZero:boolean,label:string):RobotPresetGroup{
  // Numeric values are accepted for old validation files and direct runtime
  // overrides; only the object form defines the three UI tiers.
  if(value===undefined||typeof value==='number')return {...fallback};
  if(!value||typeof value!=='object')throw new Error(`${label} must contain small, medium, large, and default.`);
  const group=value as Record<string,unknown>,result={} as RobotPresetGroup;
  for(const name of ROBOT_PRESET_NAMES){
    const number=group[name];
    if(typeof number!=='number'||!Number.isFinite(number)||(allowZero?number<0:number<=0))throw new Error(`Invalid ${label}.${name}.`);
    result[name]=number;
  }
  if(typeof group.default!=='string'||!ROBOT_PRESET_NAMES.includes(group.default as RobotPresetName))throw new Error(`Invalid ${label}.default.`);
  result.default=group.default as RobotPresetName;
  return result;
}
export function readRobotControlPresets(value:unknown):RobotControlPresets{
  if(!value||typeof value!=='object')throw new Error('robotInputParameter.json must contain an object.');
  const raw=value as Record<string,unknown>;
  return {
    viscosity:presetGroup(raw.viscosity, FALLBACK_PRESETS.viscosity,false,'viscosity'),
    surfaceTension:presetGroup(raw.surfaceTension,FALLBACK_PRESETS.surfaceTension,true,'surfaceTension'),
  };
}
export function readRobotParameters(value:unknown):RobotParameters{
  if(!value||typeof value!=='object')throw new Error('robotInputParameter.json must contain an object.');
  const raw=value as Record<string,unknown>,presets=readRobotControlPresets(value);
  const p={timeRefinement:4,...raw,
    viscosity:typeof raw.viscosity==='number'?raw.viscosity:presets.viscosity[presets.viscosity.default],
    surfaceTension:typeof raw.surfaceTension==='number'?raw.surfaceTension:presets.surfaceTension[presets.surfaceTension.default],
  } as unknown as RobotParameters;
  const keys=['gridScale','viscosity','surfaceTension','gravity','smagorinsky','waterLevel','robotContactAngle','wallContactAngle','secondsPerStep','durationSeconds','stepsPerBatch','motionSpeed','timeRefinement','routine'] as const;
  for(const key of keys)if(typeof p[key]!=='number'||!Number.isFinite(p[key]))throw new Error(`Invalid robot parameter: ${key}.`);
  if(![.375,.4375,.5625].includes(p.gridScale))throw new Error('Robot gridScale must be 0.375, 0.4375, or 0.5625.');
  if(p.viscosity<=0||p.surfaceTension<0||p.gravity>=0||p.smagorinsky<0)throw new Error('Invalid fluid parameters. Gravity must point down.');
  if(p.waterLevel<4||p.waterLevel>ROBOT_MAX_WATER_LEVEL)throw new Error(`Water level must be between 4 and ${ROBOT_MAX_WATER_LEVEL.toFixed(3)} reference lattice cells.`);
  if([p.robotContactAngle,p.wallContactAngle].some(x=>x<0||x>180))throw new Error('Contact angles must be 0–180 degrees.');
  if(p.secondsPerStep<=0||p.secondsPerStep>.002||p.durationSeconds<=0||p.durationSeconds>120)throw new Error('Invalid simulation time mapping or duration.');
  if(!Number.isInteger(p.stepsPerBatch)||p.stepsPerBatch<1||p.stepsPerBatch>128)throw new Error('stepsPerBatch must be an integer from 1 to 128.');
  if(p.motionSpeed<0.5||p.motionSpeed>4)throw new Error('motionSpeed must be between 0.5 and 4.');
  if(!Number.isInteger(p.timeRefinement)||p.timeRefinement<4||p.timeRefinement>32)throw new Error('timeRefinement must be an integer from 4 to 32.');
  if(!Number.isInteger(p.routine)||p.routine<1||p.routine>6)throw new Error('routine must be an integer from 1 to 6.');
  return Object.fromEntries(keys.map(key=>[key,p[key]])) as unknown as RobotParameters;
}
/** Acoustic refinement: dx and dt decrease together; dimensionless wave speed stays fixed. */
export function robotFluidConfig(p:RobotParameters):FluidConfig{
  const scale=p.gridScale,q=robotRefinement(p);
  return {nx:Math.round(ROBOT_DOMAIN[0]*scale),ny:Math.round(ROBOT_DOMAIN[1]*scale),nz:Math.round(ROBOT_DOMAIN[2]*scale),scale,
    nu:p.viscosity*scale/q,sigma:p.surfaceTension*scale/(q*q),gravity:p.gravity/(scale*q*q),
    particleRadius:4*ROBOT_GEOMETRY_SCALE*scale,thetaP:p.robotContactAngle,thetaS:p.wallContactAngle,smagorinsky:p.smagorinsky,
    scene:'robot-pool',waterLevel:p.waterLevel};
}
export const robotRefinement=(p:RobotParameters)=>p.timeRefinement*Math.max(1,p.motionSpeed*ROBOT_TIME_SCALE);
export const robotTimeStep=(p:RobotParameters)=>p.secondsPerStep/(p.gridScale*robotRefinement(p));

