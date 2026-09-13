/** A visual sample at a physical time in seconds. Physics remains in FP32. */
export interface PlaybackFrame {
  time:number;
  phi:Uint8Array;
  particles:Float32Array;
  orientations:Float32Array;
}

/** Interpolate visible body state without advancing the fluid solver. */
export function interpolateParticles(a:PlaybackFrame,b:PlaybackFrame,alpha:number){
  if(a.particles.length!==b.particles.length||a.orientations.length!==b.orientations.length
    ||a.particles.length%12!==0||a.orientations.length!==a.particles.length/3)
    throw new Error('Playback frames must contain the same particle layout.');
  const t=Math.max(0,Math.min(1,alpha));
  const particles=new Float32Array(a.particles.length),orientations=new Float32Array(a.orientations.length);
  for(let i=0;i<particles.length;i++)particles[i]=a.particles[i]+t*(b.particles[i]-a.particles[i]);
  for(let i=0;i<orientations.length;i+=4){
    let dot=0;for(let j=0;j<4;j++)dot+=a.orientations[i+j]*b.orientations[i+j];
    const sign=dot<0?-1:1;
    let norm=0;
    for(let j=0;j<4;j++){
      const q=a.orientations[i+j]*(1-t)+sign*b.orientations[i+j]*t;
      orientations[i+j]=q;norm+=q*q;
    }
    if(norm>1e-16){const inverse=1/Math.sqrt(norm);for(let j=0;j<4;j++)orientations[i+j]*=inverse;}
    else orientations[i+3]=1;
  }
  return{particles,orientations};
}
