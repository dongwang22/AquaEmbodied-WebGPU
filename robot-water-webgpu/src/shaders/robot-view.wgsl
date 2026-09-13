struct View {eye:vec4<f32>,forward:vec4<f32>,right:vec4<f32>,up:vec4<f32>,grid:vec4<f32>,style:vec4<f32>}
struct Body {position:vec4<f32>,velocity:vec4<f32>,omega:vec4<f32>,axis:vec4<f32>}
@group(0) @binding(0) var<uniform> view:View;
@group(0) @binding(1) var<storage,read> bodies:array<Body>;
@group(0) @binding(2) var volume:texture_3d<f32>;
@group(0) @binding(3) var filtering:sampler;
@group(0) @binding(4) var<storage,read> groups:array<vec4<f32>>;
@group(0) @binding(5) var nextVolume:texture_3d<f32>;
struct Vertex {@builtin(position) position:vec4<f32>,@location(0) uv:vec2<f32>}
@vertex fn vertex(@builtin(vertex_index) id:u32)->Vertex{
 let q=array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));return Vertex(vec4(q[id],0.0,1.0),q[id]);
}
fn interval(ro:vec3<f32>,rd:vec3<f32>,lo:vec3<f32>,hi:vec3<f32>)->vec2<f32>{
 let inv=1.0/select(rd,vec3(1e-8),abs(rd)<vec3(1e-8));let a=(lo-ro)*inv;let b=(hi-ro)*inv;
 let mn=min(a,b);let mx=max(a,b);return vec2(max(max(mn.x,mn.y),mn.z),min(min(mx.x,mx.y),mx.z));
}
fn field(q:vec3<f32>)->f32{
 if(any(q<vec3(0.0))||any(q>view.grid.xyz/view.grid.w)){return 0.0;}
 let uv=(q*view.grid.w+0.5)/view.grid.xyz;let value=textureSampleLevel(volume,filtering,uv,0.0).x;
 if(view.style.y>0.0){return mix(value,textureSampleLevel(nextVolume,filtering,uv,0.0).x,view.style.y);}return value;
}
fn waterNormal(q:vec3<f32>,ray:vec3<f32>)->vec3<f32>{
 let h=.8/view.grid.w;let g=vec3(field(q+vec3(h,0,0))-field(q-vec3(h,0,0)),
 field(q+vec3(0,h,0))-field(q-vec3(0,h,0)),field(q+vec3(0,0,h))-field(q-vec3(0,0,h)));
 if(length(g)<1e-5){return -ray;}return normalize(-g);
}
fn sphereHit(ro:vec3<f32>,rd:vec3<f32>,center:vec3<f32>,radius:f32)->f32{
 let oc=ro-center;let b=dot(oc,rd);let d=b*b-dot(oc,oc)+radius*radius;
 if(d<0.0){return 1e20;}let t=-b-sqrt(d);if(t>0.0){return t;}let exit=-b+sqrt(d);return select(1e20,exit,exit>0.0);
}
fn capsuleHit(ro:vec3<f32>,rd:vec3<f32>,body:Body)->f32{
 let center=body.position.xyz;let radius=body.position.w;let axis=body.axis.xyz;let h=body.velocity.w;
 if(h<1e-5){return sphereHit(ro,rd,center,radius);}
 let oc=ro-center;let dz=dot(rd,axis);let oz=dot(oc,axis);
 let a=1.0-dz*dz;let b=dot(oc,rd)-oz*dz;let c=dot(oc,oc)-oz*oz-radius*radius;
 let discriminant=b*b-a*c;var best=1e20;
 if(a>1e-7&&discriminant>=0.0){
  let root=sqrt(discriminant);let roots=vec2((-b-root)/a,(-b+root)/a);
  for(var k=0u;k<2u;k++){let t=roots[k];if(t>0.0&&abs(oz+t*dz)<=h){best=min(best,t);}}
 }
 // Restrict end-sphere intersections to the exterior hemispheres.
 for(var side=-1;side<=1;side+=2){
  let end=center+f32(side)*h*axis;let t=sphereHit(ro,rd,end,radius);
  if(dot(ro+rd*t-center,axis)*f32(side)>=h){best=min(best,t);}
 }
 return best;
}
@fragment fn fragment(in:Vertex)->@location(0) vec4<f32>{
 let ro=view.eye.xyz;let rd=normalize(view.forward.xyz+in.uv.x*view.right.xyz*view.forward.w*view.right.w+in.uv.y*view.up.xyz*view.forward.w);
 let size=view.grid.xyz/view.grid.w;let light=normalize(vec3(0.6,-0.5,1.0));
 var background=mix(vec3(.026,.044,.065),vec3(.075,.12,.17),clamp(.5+.5*in.uv.y,0.0,1.0));
 var depth=1e20;var normal=vec3(0.0,0.0,1.0);var base=vec3(.2);var material=0u;
 // The floor and ceiling remain solid computational walls. Lateral faces are periodic.
 let floorZ=.5/view.grid.w;
 if(rd.z< -1e-7){let t=(floorZ-ro.z)/rd.z;let q=ro+rd*t;
  if(t>0.0&&all(q.xy>=vec2(0.0))&&all(q.xy<=size.xy)){
   depth=t;material=1u;let line=min(abs(fract(q.x/12.0)-.5),abs(fract(q.y/12.0)-.5));
   base=mix(vec3(.11,.19,.22),vec3(.22,.32,.34),smoothstep(.008,.025,line));
  }
 }
 for(var group=0u;group<arrayLength(&groups)/2u;group++){
  let low=groups[group*2u];let high=groups[group*2u+1u];let span=interval(ro,rd,low.xyz,high.xyz);
  if(span.y<max(0.0,span.x)||span.x>depth){continue;}
  for(var i=u32(low.w);i<min(u32(high.w),arrayLength(&bodies));i++){
   let body=bodies[i];let t=capsuleHit(ro,rd,body);
   if(t>0.0&&t<depth){
    depth=t;material=2u;let offset=ro+rd*t-body.position.xyz;
    normal=normalize(offset-body.axis.xyz*clamp(dot(offset,body.axis.xyz),-body.velocity.w,body.velocity.w));
    base=vec3(.82,.84,.87);
    if(body.omega.w>.5){base=vec3(.28,.34,.42);}
    if(body.omega.w>1.5){base=vec3(.06,.72,.86);}
    if(body.omega.w<.5&&abs(dot(normal,body.axis.xyz))>.88){base=vec3(.18,.28,.34);}
   }
  }
 }
 if(material>0u){let diffuse=max(0.0,dot(normal,light));let spec=pow(max(0.0,dot(normal,normalize(light-rd))),40.0);
  background=base*(.38+.62*diffuse)+select(.025,.22,material==2u)*spec;
 }
 let domain=interval(ro,rd,vec3(0.0),size);let start=max(0.0,domain.x);let end=min(depth,domain.y);
 let delta=.9/view.grid.w;var first=-1.0;var thickness=0.0;var t=start;
 if(end>start){for(var k=0u;k<512u;k++){
  if(t>end){break;}let value=field(ro+rd*t);
  if(value>=.45){
   if(first<0.0){var lo=max(start,t-delta);var hi=t;
    for(var refine=0u;refine<5u;refine++){let mid=.5*(lo+hi);if(field(ro+rd*mid)>=.45){hi=mid;}else{lo=mid;}}first=hi;
   }
   thickness+=value*delta;
  }t+=delta;
 }}
 if(first>=0.0){
  let q=ro+rd*first;let n=waterNormal(q,rd);let fresnel=pow(1.0-max(0.0,dot(n,-rd)),4.0);
  let diffuse=max(0.0,dot(n,light));let spec=pow(max(0.0,dot(n,normalize(light-rd))),85.0);
  let water=vec3(.025,.34,.47)*(.65+.35*diffuse)+vec3(.19,.32,.36)*fresnel+vec3(.8)*spec;
  let alpha=clamp(view.style.x*(.35+.65*(1.0-exp(-thickness/35.0)))+.25*fresnel,0.0,.95);
  background=mix(background,water,alpha);
 }
 return vec4(pow(max(background,vec3(0.0)),vec3(.85)),1.0);
}

