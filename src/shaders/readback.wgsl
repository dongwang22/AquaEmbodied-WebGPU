@compute @workgroup_size(64)
fn gatherSlice(@builtin(global_invocation_id) id:vec3<u32>){
 let s=linearInvocation(id);if(s>=p.grid.x*p.grid.z){return;}
 let n=s%p.grid.x+(p.control.z+(s/p.grid.x)*p.grid.y)*p.grid.x;
 output[2u*s]=vec4(cells[n].rho,cells[n].phi,bitcast<f32>(cells[n].flag),cells[n].mass);
 output[2u*s+1u]=vec4(velocity(n),cells[n].excess);
}
@compute @workgroup_size(64)
fn gatherAll(@builtin(global_invocation_id) id:vec3<u32>){
 let n=linearInvocation(id);if(n>=p.grid.w){return;}
 output[2u*n]=vec4(cells[n].rho,cells[n].phi,bitcast<f32>(cells[n].flag),cells[n].mass);
 output[2u*n+1u]=vec4(velocity(n),cells[n].excess);
}
// Read each of the six external faces; edges/corners deliberately appear on
// each adjoining face so no face can pass by borrowing another face's count.
@compute @workgroup_size(64)
fn gatherBoundary(@builtin(global_invocation_id) id:vec3<u32>){
 let k=linearInvocation(id);let yz=p.grid.y*p.grid.z;let xz=p.grid.x*p.grid.z;let xy=p.grid.x*p.grid.y;
 if(k>=2u*(yz+xz+xy)){return;}
 var q=vec3<i32>(0);
 if(k<2u*yz){let j=k%yz;q=vec3<i32>(i32((k/yz)*(p.grid.x-1u)),i32(j%p.grid.y),i32(j/p.grid.y));}
 else if(k<2u*(yz+xz)){let j=(k-2u*yz)%xz;q=vec3<i32>(i32(j%p.grid.x),i32(((k-2u*yz)/xz)*(p.grid.y-1u)),i32(j/p.grid.x));}
 else{let j=(k-2u*(yz+xz))%xy;q=vec3<i32>(i32(j%p.grid.x),i32(j/p.grid.x),i32(((k-2u*(yz+xz))/xy)*(p.grid.z-1u)));}
 let n=index(q);
 output[2u*k]=vec4(bitcast<f32>(cells[n].flag),bitcast<f32>(cells[n].owner),cells[n].mass,abs(cells[n].excess)+abs(cells[n].debt));
 output[2u*k+1u]=vec4(velocity(n),0.0);
}
// One compact record per workgroup: totals, extrema, flag counts and bad
// cells. Pending excess is stored as a total, independent of recipient count.
var<workgroup> sums:array<vec4<f32>,64>;
var<workgroup> extrema:array<vec4<f32>,64>;
var<workgroup> counts:array<vec4<f32>,64>;
var<workgroup> eddy:array<vec2<f32>,64>;
@compute @workgroup_size(64)
fn reduceDiagnostics(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) wid:vec3<u32>){
 let n=linearInvocation(id);var a=vec4(0.0);var b=vec4(1e30,-1e30,0.0,0.0);var c=vec4(0.0);
 var les=vec2(0.0);
 if(n<p.grid.w){
  let flag=cells[n].flag;let rho=cells[n].rho;let u=velocity(n);
  a=vec4(cells[n].mass,cells[n].excess,cells[n].debt,select(0.0,cells[n].phi,flag!=S));
  if(flag==F||flag==I){
   b=vec4(rho,rho,length(u),0.0);les=vec2(cells[n].eddyViscosity);
   if(!(les.x>=0.0&&les.x<1e30)){b.w=1.0;}
  }
  if(!(rho>=0.1&&rho<=10.0)||!(abs(cells[n].mass)<1e30)||!(abs(cells[n].phi)<1e30)||!all(abs(u)<vec3(1e30))||!(abs(cells[n].excess)<1e30)||!(abs(cells[n].debt)<1e30)){b.w=1.0;}
  c=vec4(select(0.0,1.0,flag==F),select(0.0,1.0,flag==I),select(0.0,1.0,flag==G),select(0.0,1.0,flag==S));
 }
 sums[lid.x]=a;extrema[lid.x]=b;counts[lid.x]=c;eddy[lid.x]=les;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){
  if(lid.x<stride){sums[lid.x]+=sums[lid.x+stride];counts[lid.x]+=counts[lid.x+stride];let r=extrema[lid.x+stride];extrema[lid.x]=vec4(min(extrema[lid.x].x,r.x),max(extrema[lid.x].y,r.y),max(extrema[lid.x].z,r.z),extrema[lid.x].w+r.w);eddy[lid.x]=vec2(max(eddy[lid.x].x,eddy[lid.x+stride].x),eddy[lid.x].y+eddy[lid.x+stride].y);}
  workgroupBarrier();
 }
 if(lid.x==0u){let group=wid.x+wid.y*65535u;if(group<(p.grid.w+63u)/64u){output[group*4u]=sums[0];output[group*4u+1u]=extrema[0];output[group*4u+2u]=counts[0];output[group*4u+3u]=vec4(eddy[0],0.0,0.0);}}
}

