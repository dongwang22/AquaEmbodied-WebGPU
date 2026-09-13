@compute @workgroup_size(64)
fn initialize(@builtin(global_invocation_id) id:vec3<u32>){
 let n=linearInvocation(id);if(n>=p.grid.w){return;}
 let q=coord(n);var flag=G;var phi=0.0;var owner=0u;
 if(terrain(q)){flag=S;phi=0.5*(1.0+cos(p.geometry.z));}
 else{
  owner=covering(n);
  if(owner>0u){flag=S;phi=0.5*(1.0+cos(p.geometry.y));}
  else if(initialFill(q)){flag=F;phi=1.0;}
  else{
   for(var d=1u;d<19u;d++){
    let j=neighbor(n,d);let a=coord(j);
    if(!terrain(a)&&initialFill(a)&&covering(j)==0u){flag=I;phi=0.5;break;}
   }
  }
 }
 let u=wallVelocity(n,owner);
 let rho=select(1.0,initialDensity(q),wet(flag));
 cells[n]=Cell(rho,select(phi*rho,0.0,flag==S),phi,0.0,u.x,u.y,u.z,flag,owner,owner,0.0,0.0);
 flags[n]=flag;
 // Initialize shifted populations from the pool's hydrostatic density.
 for(var d=0u;d<19u;d++){
  // Odd storage planes belong to the neighbouring cell in Esoteric Pull.
  // This yields a local hydrostatic equilibrium for every incoming direction.
  let sourceCoord=select(q,q-C[d],(d&1u)==1u);
  popSet(n,d,weight(d)*(initialDensity(sourceCoord)-1.0));
 }
}

@compute @workgroup_size(64)
fn snapshotFlags(@builtin(global_invocation_id) id:vec3<u32>){
 let n=linearInvocation(id);if(n<p.grid.w){flags[n]=cells[n].flag;}
}

// Target ownership is computed separately so mass can be sent exclusively
// to neighbours which remain outside every moving particle.
@compute @workgroup_size(64)
fn mapParticles(@builtin(global_invocation_id) id:vec3<u32>){
 let n=linearInvocation(id);if(n>=p.grid.w){return;}
 cells[n].nextOwner=select(covering(n),0u,terrain(coord(n)));
}

fn applyParticleOwner(n:u32,owner:u32){
 let old=cells[n].owner;
 if(owner>0u){
  if(old==0u){
   // Keep a total until the final repaired map determines the recipients.
   // Counting here undercounts newly created interface cells and creates mass.
   cells[n].excess+=cells[n].mass+cells[n].debt;cells[n].debt=0.0;
   cells[n].mass=0.0;
  }
  cells[n].flag=S;cells[n].phi=0.5*(1.0+cos(p.geometry.y));cells[n].rho=1.0;
  cells[n].eddyViscosity=0.0;
  setVelocity(n,wallVelocity(n,owner));
 }else if(old>0u){
  cells[n].flag=UNCOVERED;cells[n].phi=0.0;cells[n].mass=cells[n].debt;cells[n].debt=0.0;
  // Pending displaced mass belongs to the cell even after the particle leaves.
  cells[n].rho=1.0;setVelocity(n,vec3(0.0));
  cells[n].eddyViscosity=0.0;
 }
 cells[n].owner=owner;
}
@compute @workgroup_size(64)
fn updateWalls(@builtin(global_invocation_id) id:vec3<u32>){
 let n=linearInvocation(id);if(n>=p.grid.w||terrain(coord(n))){return;}
 applyParticleOwner(n,cells[n].nextOwner);
}
@compute @workgroup_size(64)
fn mapAndUpdateWalls(@builtin(global_invocation_id) id:vec3<u32>){
 let n=linearInvocation(id);if(n>=p.grid.w){return;}
 // Static terrain is immutable; only particle-owned solids can uncover.
 if(cells[n].flag!=S||cells[n].owner!=0u){applyParticleOwner(n,covering(n));}
 flags[n]=cells[n].flag;
}

// Separate dispatch: read the immutable post-map flags and only read fluid
// fields from surviving wet cells. Adjacent uncovered cells cannot race or
// mistake one another for gas. Account for refilled liquid with signed excess.
@compute @workgroup_size(64)
fn refillUncovered(@builtin(global_invocation_id) id:vec3<u32>){
 let n=linearInvocation(id);if(n>=p.grid.w||flags[n]!=UNCOVERED){return;}
 var sum=vec4(0.0);var fill=0.0;var count=0.0;var hasGas=false;var hasInterface=false;
 for(var d=1u;d<27u;d++){
  let j=neighbor(n,d);let flag=flags[j];
  hasGas=hasGas||flag==G;hasInterface=hasInterface||flag==I;
  if(flag==F||flag==I){sum+=vec4(velocity(j),cells[j].rho);fill+=cells[j].phi;count+=1.0;}
 }
 var flag=G;var phi=0.0;var avg=vec4(0.0,0.0,0.0,1.0);
 if(count>0.0){
  avg=sum/count;
  if(!hasGas&&!hasInterface){flag=F;phi=1.0;}
  else{flag=I;phi=clamp(fill/count,0.0,1.0);}
 }
 let mass=phi*avg.w;
 cells[n].excess+=cells[n].mass-mass;
 cells[n].mass=mass;cells[n].rho=avg.w;cells[n].phi=phi;cells[n].flag=flag;
 setVelocity(n,avg.xyz);
 if(flag!=G){for(var d=0u;d<19u;d++){storeF(n,d,p.control.x^1u,equilibrium(d,avg.w,avg.xyz));}}
}

@compute @workgroup_size(64)
fn repairInterface(@builtin(global_invocation_id) id:vec3<u32>){
 let n=linearInvocation(id);if(n>=p.grid.w){return;}
 if(optimizedKernels){let q=vec3<f32>(coord(n));if(any(q<p.particleMin.xyz-2.0)||any(q>p.particleMax.xyz+2.0)){return;}}
 let flag=flags[n];if(flag!=G&&flag!=F){return;}
 var hasF=false;var hasG=false;
 for(var d=1u;d<19u;d++){let f=flags[neighbor(n,d)];hasF=hasF||f==F;hasG=hasG||f==G;}
 if(flag==G&&hasF){
  cells[n].flag=I;cells[n].phi=0.0;
  let avg=averageWet(n);
  // The map is rebuilt before this step; populations still have the
  // previous collision's parity (t-1), equivalent to CUDA post-step repair.
  for(var d=0u;d<19u;d++){storeF(n,d,p.control.x^1u,equilibrium(d,avg.w,avg.xyz));}
 }else if(flag==F&&hasG){cells[n].flag=I;cells[n].phi=clamp(cells[n].mass/max(cells[n].rho,1e-8),0.0,1.0);}
}

