export type PopulationPrecision='fp32'|'fp16s';
export interface PlaneGroup {start:number;count:number}
/** FluidX3D FP16S: store shifted distributions multiplied by 2^15 in IEEE half.
 * Collision, force, mass and all reductions continue to execute in FP32. */
export function populationSource(groups:PlaneGroup[],precision:PopulationPrecision){
  const half=precision==='fp16s';
  const declarations=groups.map((_,i)=>`@group(0) @binding(${6+i}) var<storage,read_write> pop${i}:array<${half?'f16':'f32'}>;`).join('\n');
  const getters=groups.map(({start,count},i)=>{
    const access=`pop${i}[(d-${start}u)*p.grid.w+n]`;
    return `${i?'else ':''}if(d<${start+count}u){return ${half?`f32(${access})*0.000030517578125`:access};}`;
  }).join('\n');
  const setters=groups.map(({start,count},i)=>`${i?'else ':''}if(d<${start+count}u){pop${i}[(d-${start}u)*p.grid.w+n]=${half?'f16(v*32768.0)':'v'};}`).join('\n');
  return`${declarations}\nfn popGet(n:u32,d:u32)->f32{${getters}\nreturn 0.0;}\nfn popSet(n:u32,d:u32,v:f32){${setters}}`;
}

