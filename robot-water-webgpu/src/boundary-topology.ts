export const FLUID_FACES=['x−','x+','y−','y+','z−','z+'] as const;
export type FluidBoundaryCondition='periodic'|'solid-wall';

/** Boundary topology shared by diagnostics and the public scene description. */
export function fluidBoundaryConditions():readonly FluidBoundaryCondition[]{
  return ['periodic','periodic','periodic','periodic','solid-wall','solid-wall'];
}
