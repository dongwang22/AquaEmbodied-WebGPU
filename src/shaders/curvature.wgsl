// PLIC cube-plane inversion and Parker-Youngs normals, following the CUDA
// reference. Degenerate normals and ill-conditioned least-squares systems
// are handled explicitly instead of allowing NaN to select a clamp bound.
fn cube(v:f32)->f32{return v*v*v;}
fn plicReduced(volume:f32,n1:f32,n2:f32,n3:f32)->f32{
 let nv=n3*volume;let n12=n1+n2;
 if(n12<=2.0*nv){return nv+0.5*n12;}
 if(n1<1e-7){return sqrt(max(0.0,2.0*n2*nv));}
 let n1sq=n1*n1;let n26=6.0*n2;let v1=n1sq/n26;
 if(v1<=nv&&nv<v1+0.5*(n2-n1)){return 0.5*(n1+sqrt(max(0.0,n1sq+8.0*n2*(nv-v1))));}
 let v6=n1*n26*nv;
 if(nv<v1){return pow(max(0.0,v6),1.0/3.0);}
 var v3=0.5*n12;
 if(n3<n12){v3=(n3*n3*(3.0*n12-n3)+n1sq*(n1-3.0*n3)+n2*n2*(n2-3.0*n3))/(n1*n26);}
 let s=n1sq+n2*n2;let vc=v6-cube(n1)-cube(n2);let case34=nv<v3;
 let a=select(0.5*(vc-cube(n3)),vc,case34);
 let b=select(0.5*(s+n3*n3),s,case34);let c=select(0.5,n12,case34);
 let t=sqrt(max(c*c-b,1e-14));
 return c-2.0*t*sin(asin(clamp((cube(c)-0.5*a-1.5*b*c)/cube(t),-1.0,1.0))/3.0);
}
fn plic(volume:f32,normal:vec3<f32>)->f32{
 let a=abs(normal);let l=a.x+a.y+a.z;
 if(l<1e-10){return 0.0;}
 let n1=min(min(a.x,a.y),a.z)/l;let n3=max(max(a.x,a.y),a.z)/l;let n2=max(0.0,1.0-n1-n3);
 let v=clamp(volume,0.0,1.0);let d=plicReduced(0.5-abs(v-0.5),n1,n2,n3);
 return l*select(d-0.5,0.5-d,v>=0.5);
}
fn curvature(n:u32,centerPhi:f32)->f32{
 var values:array<f32,27>;values[0]=centerPhi;var gradient=vec3(0.0);
 for(var d=1u;d<27u;d++){
  let j=neighbor(n,d);values[d]=cells[j].phi;
  if(cells[j].flag==S){
   // A solid's stored phi encodes wetting, not a liquid/gas sample. Only use
   // that ghost value at an actual gas-contacting wall. Submerged solids use
   // zero normal extrapolation and cannot create a fictitious contact line.
   var touchesGas=false;
   for(var k=1u;k<27u;k++){touchesGas=touchesGas||cells[neighbor(j,k)].flag==G;}
   if(!touchesGas){values[d]=centerPhi;}
  }
  let c=vec3<f32>(C[d]);let count=abs(c.x)+abs(c.y)+abs(c.z);
  let w=select(select(1.0,2.0,count==2.0),4.0,count==1.0);
  gradient-=w*c*values[d];
 }
 if(dot(gradient,gradient)<1e-12){return 0.0;}
 let bz=normalize(gradient);let reference=vec3(0.562709,0.32704452,0.75921047);
 let by=safeNormal(cross(bz,reference));let bx=cross(by,bz);
 let center=plic(centerPhi,bz);
 var matrix:array<f32,25>;var rhs:array<f32,5>;var count=0u;
 for(var d=1u;d<27u;d++){
  // Fit real interface points only; solid wetting ghosts have no PLIC surface.
  if(cells[neighbor(n,d)].flag!=I){continue;}
  let v=values[d];if(v<=0.0||v>=1.0){continue;}
  let c=vec3<f32>(C[d]);let x=dot(c,bx);let y=dot(c,by);let z=dot(c,bz)+plic(v,bz)-center;
  let basis=array<f32,5>(x*x,y*y,x*y,x,y);
  for(var a=0u;a<5u;a++){rhs[a]+=basis[a]*z;for(var b=0u;b<5u;b++){matrix[a*5u+b]+=basis[a]*basis[b];}}
  count++;
 }
 if(count<3u){return 0.0;}
 // Weak Tikhonov regularization also makes flat/collinear contact-line fits
 // finite. Partial pivoting is more robust than the original unpivoted LU.
 for(var a=0u;a<5u;a++){matrix[a*5u+a]+=1e-6;}
 for(var a=0u;a<5u;a++){
  var pivot=a;var best=abs(matrix[a*5u+a]);
  for(var row=a+1u;row<5u;row++){if(abs(matrix[row*5u+a])>best){best=abs(matrix[row*5u+a]);pivot=row;}}
  if(best<1e-10){return 0.0;}
  if(pivot!=a){for(var b=0u;b<5u;b++){let tmp=matrix[a*5u+b];matrix[a*5u+b]=matrix[pivot*5u+b];matrix[pivot*5u+b]=tmp;}let tmp=rhs[a];rhs[a]=rhs[pivot];rhs[pivot]=tmp;}
  let diag=matrix[a*5u+a];for(var b=a;b<5u;b++){matrix[a*5u+b]/=diag;}rhs[a]/=diag;
  for(var row=0u;row<5u;row++){if(row==a){continue;}let f=matrix[row*5u+a];for(var b=a;b<5u;b++){matrix[row*5u+b]-=f*matrix[a*5u+b];}rhs[row]-=f*rhs[a];}
 }
 let a=rhs[0];let b=rhs[1];let c=rhs[2];let h=rhs[3];let j=rhs[4];
 return clamp((a*(j*j+1.0)+b*(h*h+1.0)-c*h*j)/pow(h*h+j*j+1.0,1.5),-1.0,1.0);
}

