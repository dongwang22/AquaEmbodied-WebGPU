/** Physical validity and FP32 representability, without empirical stability bounds. */
export function validateFluidParameters(p:{nu:number;sigma:number;gravity:number;smagorinsky:number;scale:number}) {
  for(const [name,value] of Object.entries(p)) {
    if(!Number.isFinite(value)||!Number.isFinite(Math.fround(value)))
      throw new Error(`${name} must be a finite value representable by GPU FP32.`);
  }
  if(!(p.nu>0)||Math.fround(p.nu)===0)throw new Error('Viscosity must be greater than zero.');
  if(p.sigma<0)throw new Error('Surface tension must not be negative.');
  if(p.smagorinsky<0)throw new Error('LES coefficient must not be negative; zero disables LES.');
  if(!(p.scale>0))throw new Error('Grid scale must be greater than zero.');
  if(!Number.isFinite(Math.fround(3*p.nu+.5))||!Number.isFinite(Math.fround(18*Math.SQRT2*p.smagorinsky**2)))
    throw new Error('Relaxation time or LES coefficient exceeds the GPU FP32 range.');
}

