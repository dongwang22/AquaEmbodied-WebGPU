// Read-only visualization copy. This never writes the solver's Cell buffer.
struct View {
 eye:vec4<f32>, forward:vec4<f32>, right:vec4<f32>, up:vec4<f32>,
 grid:vec4<f32>, style:vec4<f32>,
}
struct Cell {
 rho:f32,mass:f32,phi:f32,excess:f32,
 ux:f32,uy:f32,uz:f32,flag:u32,
 owner:u32,nextOwner:u32,debt:f32,eddyViscosity:f32,
}
@group(0) @binding(0) var<uniform> view:View;
@group(0) @binding(1) var<storage,read> cells:array<Cell>;
@group(0) @binding(2) var volume:texture_storage_3d<rgba16float,write>;
@compute @workgroup_size(4,4,4)
fn copyVolume(@builtin(global_invocation_id) id:vec3<u32>){
 let size=vec3<u32>(view.grid.xyz);if(any(id>=size)){return;}
 let n=id.x+(id.y+id.z*size.y)*size.x;
 let c=cells[n];let phi=select(clamp(c.phi,0.0,1.0),0.0,c.flag==1u);
 textureStore(volume,vec3<i32>(id),vec4(phi,length(vec3(c.ux,c.uy,c.uz)),c.rho,0.0));
}

