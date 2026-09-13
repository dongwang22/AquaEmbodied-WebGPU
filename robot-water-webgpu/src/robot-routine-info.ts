import {ROBOT_AUTHORED_SECONDS,ROBOT_TIME_SCALE,type RoutineId} from './robot-motion';
type Stage={start:number;end:number;label:string};
/** User-authored intervals; gaps between stages are one-second transitions. */
export const ROUTINE_STAGES:Record<RoutineId,readonly Stage[]>={
  1:[{start:0,end:5,label:'3 double side slaps'},{start:6,end:12,label:'4 overhead forward slams'},{start:13,end:20,label:'5 outward scoops'},{start:21,end:30,label:'Alternating side slaps and salute'}],
  2:[{start:0,end:6,label:'4 alternating rear leg sweeps'},{start:7,end:14,label:'6 paired kicks from back support'},{start:15,end:22,label:'4 paired left–right sweep cycles'},{start:23,end:30,label:'4 seated V-leg slaps and hold'}],
  3:[{start:0,end:7,label:'4 right front kicks'},{start:8,end:14,label:'4 left front kicks'},{start:15,end:22,label:'6 alternating outward kicks'},{start:23,end:30,label:'3 overhead scoops and salute'}],
  4:[{start:0,end:8,label:'5 forward chops in horse stance'},{start:9,end:16,label:'5 double outward chops'},{start:17,end:24,label:'8 accelerating side punches'},{start:25,end:30,label:'3 side slaps and outward throw'}],
  5:[{start:0,end:7,label:'4 full longitudinal rolls'},{start:8,end:15,label:'5 paired rear leg slaps'},{start:16,end:23,label:'6 paired pendulum slaps'},{start:24,end:30,label:'3 seated V-leg slaps and hold'}],
  6:[{start:0,end:6,label:'3 closed-leg jump landings'},{start:7,end:14,label:'4 wide jump and forward slams'},{start:15,end:22,label:'6 alternating lateral jumps'},{start:23,end:30,label:'High jump, splash and kneeling salute'}],
};
export function routineStageLabel(id:RoutineId,time:number):string{
  if(time>=ROBOT_AUTHORED_SECONDS)return 'Final pose';
  const stages=ROUTINE_STAGES[id],stage=stages.find(s=>time>=s.start&&time<=s.end);
  const seconds=(value:number)=>Number((value/ROBOT_TIME_SCALE).toFixed(2));
  if(stage)return `${seconds(stage.start)}–${seconds(stage.end)} s · ${stage.label}`;
  const next=stages.find(s=>s.start>time);return next?`Transition · ${next.label}`:'Final pose';
}

