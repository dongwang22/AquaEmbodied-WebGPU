// Distribute TOTAL excess against a single, immutable, post-repair flag map.
// Three passes keep every source total and recipient count readable until
// every destination has gathered its share. A source with no recipient keeps
// its total for a later step. nextOwner is free scratch after updateWalls.
@compute @workgroup_size(64)
fn countMassRecipients(@builtin(global_invocation_id) id:vec3<u32>){
 let n=linearInvocation(id);if(n>=p.grid.w){return;}
 if(optimizedKernels){flags[n]=cells[n].flag;}
 var count=0u;
 if(cells[n].excess!=0.0){for(var d=1u;d<19u;d++){let j=neighbor(n,d);var flag=0u;if(optimizedKernels){flag=cells[j].flag;}else{flag=flags[j];}if(wet(flag)){count++;}}}
 cells[n].nextOwner=count;
}
@compute @workgroup_size(64)
fn distributeMass(@builtin(global_invocation_id) id:vec3<u32>){
 let n=linearInvocation(id);if(n>=p.grid.w||!wet(flags[n])){return;}
 var mass=cells[n].mass;
 for(var d=1u;d<19u;d++){
  let j=neighbor(n,d);let count=cells[j].nextOwner;
  if(count>0u){mass+=cells[j].excess/f32(count);}
 }
 cells[n].mass=mass;
 // Every interface link uses the SAME pre-exchange fill fractions at both
 // ends. Mixing a recalculated local phi with an old neighbour phi leaks mass.
 cells[n].phi=select(clamp(mass/max(cells[n].rho,1e-8),0.0,1.0),1.0,flags[n]==F);
}
@compute @workgroup_size(64)
fn clearDistributedMass(@builtin(global_invocation_id) id:vec3<u32>){
 let n=linearInvocation(id);if(n>=p.grid.w){return;}
 if(cells[n].nextOwner>0u){cells[n].excess=0.0;}
}

@compute @workgroup_size(64)
fn surfaceExchange(@builtin(global_invocation_id) id:vec3<u32>){
 let n=linearInvocation(id);if(n>=p.grid.w){return;}
 if(optimizedKernels&&cells[n].nextOwner>0u){cells[n].excess=0.0;}
 let flag=flags[n];if(flag!=F&&flag!=I){return;}
 let t=p.control.x;var fin:array<f32,19>;var fout:array<f32,19>;
 var mass=cells[n].mass;var rho=1.0;var momentum=vec3(0.0);
 for(var d=0u;d<19u;d++){
  fin[d]=incoming(n,d,t);fout[d]=outgoing(n,d,t);
  rho+=fout[d];momentum+=vec3<f32>(C[d])*fout[d];
 }
 if(flag==F){for(var d=1u;d<19u;d++){
  let f=flags[neighbor(n,d)];
  // Newly solid links still contain the previous fluid's populations.
  // They must not exchange liquid mass through the new particle surface.
  if(f==F||f==I){mass+=fin[opposite(d)]-fout[d];}
 }}
 else{
  let phi=cells[n].phi;
  var kappa=0.0;if(p.physical.y!=0.0){kappa=curvature(n,phi);}
  let ug=clamp(momentum/max(rho,1e-8)+vec3(0.0,0.0,0.5*p.physical.z/max(rho,1e-8)),vec3(-0.57735027),vec3(0.57735027));
  let gasRho=1.0-6.0*p.physical.y*kappa;
  for(var d=1u;d<19u;d++){
   let j=neighbor(n,d);let f=flags[j];let opp=opposite(d);
   if(f==F){mass+=fin[opp]-fout[d];}
   else if(f==I){mass+=0.5*(cells[j].phi+phi)*(fin[opp]-fout[d]);}
   else if(f==G){reconstruct(n,opp,t,equilibrium(d,gasRho,ug)+equilibrium(opp,gasRho,ug)-fout[d]);}
  }
 }
 cells[n].mass=mass;
}

@compute @workgroup_size(64)
fn collide(@builtin(global_invocation_id) id:vec3<u32>){
 let n=linearInvocation(id);if(n>=p.grid.w){return;}
 let flag=flags[n];if(flag!=F&&flag!=I){return;}
 var values:array<f32,19>;var rho=1.0;var momentum=vec3(0.0);var hasF=false;var hasG=false;
 for(var d=0u;d<19u;d++){
  var value=incoming(n,d,p.control.x);
  if(d>0u){
   let j=neighbor(n,opposite(d));let f=flags[j];hasF=hasF||f==F;hasG=hasG||f==G;
   if(f==S){
    // Reflect the outgoing population using today's solid map. Its slot is
    // an input slot owned by the non-colliding solid neighbour, so no other
    // fluid invocation can overwrite it while this collision reads it.
    value=outgoing(n,opposite(d),p.control.x)
     +6.0*weight(d)*cells[n].rho*dot(vec3<f32>(C[d]),velocity(j));
   }
  }
  values[d]=value;rho+=value;momentum+=vec3<f32>(C[d])*value;
 }
 if(flag==I){
  if(cells[n].mass>rho||!hasG){cells[n].flag=IF;}
  else if(cells[n].mass<0.0||!hasF){cells[n].flag=IG;}
 }
 // Preserve the reference clamp; invalid densities are reported, not reset.
 let force=vec3(0.0,0.0,p.physical.z);
 let u=clamp((momentum+0.5*force)/rho,vec3(-0.57735027),vec3(0.57735027));
 cells[n].rho=rho;setVelocity(n,u);
 let tau0=3.0*p.physical.x+0.5;
 var omega=1.0/tau0;
 var eq:array<f32,19>;
 for(var d=0u;d<19u;d++){eq[d]=equilibrium(d,rho,u);}
 cells[n].eddyViscosity=0.0;
 if(p.geometry.w>0.0){
  // Smagorinsky-Lilly, following FluidX3D-1.0 src/kernel.cpp SUBGRID.
  // Use incoming, wall-corrected shifted f and the force-corrected equilibrium,
  // exactly as the reference. Delta=1 lattice cell at every grid resolution.
  var diagonal=vec3(0.0);var offDiagonal=vec3(0.0);
  for(var d=1u;d<19u;d++){
   let c=vec3<f32>(C[d]);let neq=values[d]-eq[d];
   diagonal+=c*c*neq;
   offDiagonal+=vec3(c.x*c.y,c.x*c.z,c.y*c.z)*neq;
  }
  let q=dot(diagonal,diagonal)+2.0*dot(offDiagonal,offDiagonal);
  let coefficient=25.45584412271571*p.geometry.w*p.geometry.w; // 18*sqrt(2)*Cs²
  omega=2.0/(tau0+sqrt(tau0*tau0+coefficient*sqrt(q)/rho));
  cells[n].eddyViscosity=max(0.0,(1.0/omega-tau0)/3.0);
 }
 for(var d=0u;d<19u;d++){
  let c=vec3<f32>(C[d]);
  let guo=weight(d)*(3.0*dot(c-u,force)+9.0*dot(c,u)*dot(c,force));
  let post=(1.0-omega)*values[d]+omega*eq[d]+(1.0-0.5*omega)*guo;
  storeF(n,d,p.control.x,post);
 }
}

// CUDA scatter writes are expressed as gather passes with snapshots. This
// removes cross-workgroup flag read/write races without changing the rules.
@compute @workgroup_size(64)
fn surfacePromote(@builtin(global_invocation_id) id:vec3<u32>){
 let n=linearInvocation(id);if(n>=p.grid.w){return;}
 let f=flags[n];if(f!=G&&f!=IG){return;}
 for(var d=1u;d<19u;d++){if(flags[neighbor(n,d)]==IF){cells[n].flag=select(GI,I,f==IG);return;}}
}
@compute @workgroup_size(64)
fn surfaceCreate(@builtin(global_invocation_id) id:vec3<u32>){
 let n=linearInvocation(id);if(n>=p.grid.w){return;}
 let f=flags[n];
 if(f==GI){let avg=averageWet(n);for(var d=0u;d<19u;d++){storeF(n,d,p.control.x,equilibrium(d,avg.w,avg.xyz));}}
 else if(f==F||f==IF){for(var d=1u;d<19u;d++){if(flags[neighbor(n,d)]==IG){cells[n].flag=I;return;}}}
}
@compute @workgroup_size(64)
fn surfaceCommit(@builtin(global_invocation_id) id:vec3<u32>){
 let n=linearInvocation(id);if(n>=p.grid.w){return;}
 let flag=flags[n];if(flag==S){return;}
 let rho=cells[n].rho;var mass=cells[n].mass;var extra=0.0;var phi=0.0;var finalFlag=flag;
 if(flag==F||flag==IF){extra=mass-rho;mass=rho;phi=1.0;finalFlag=F;}
 else if(flag==I||flag==GI){extra=select(select(0.0,mass,mass<0.0),mass-rho,mass>rho);mass=clamp(mass,0.0,max(rho,0.0));phi=clamp(mass/max(rho,1e-8),0.0,1.0);finalFlag=I;}
 else if(flag==G||flag==IG){extra=mass;mass=0.0;finalFlag=G;}
 // Unpaid totals survive until a later repaired map supplies wet recipients.
 cells[n].excess+=extra;
 cells[n].mass=mass;cells[n].phi=phi;cells[n].flag=finalFlag;
 if(optimizedKernels){flags[n]=finalFlag;}
}

