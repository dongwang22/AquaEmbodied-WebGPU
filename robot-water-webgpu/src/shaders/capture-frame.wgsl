// Visual recording only. Four volume fractions share each output word;
// no solver field or population is modified by this kernel.
struct Cell {
 rho:f32,mass:f32,phi:f32,excess:f32,
 ux:f32,uy:f32,uz:f32,flag:u32,
 owner:u32,nextOwner:u32,debt:f32,eddyViscosity:f32,
}
@group(0) @binding(0) var<storage,read> cells:array<Cell>;
@group(0) @binding(1) var<storage,read_write> packed:array<u32>;
@compute @workgroup_size(128)
fn captureFrame(@builtin(global_invocation_id) id:vec3<u32>){
 let word=id.x;if(word>=arrayLength(&packed)){return;}
 var value=0u;
 for(var lane=0u;lane<4u;lane++){
  let n=4u*word+lane;
  if(n<arrayLength(&cells)){
   let cell=cells[n];let phi=select(clamp(cell.phi,0.0,1.0),0.0,cell.flag==1u);
   value|=u32(round(255.0*phi))<<(8u*lane);
  }
 }
 packed[word]=value;
}

