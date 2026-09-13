# Architecture

## Simulation pipeline

Each displayed state is produced by an ordered GPU pipeline:

1. The choreography evaluates a connected articulated pose.
2. Capsule endpoints, axes and material-point velocities are packed into a GPU body
   buffer.
3. Changed solid cells are mapped and newly uncovered fluid cells are reconstructed.
4. Free-surface mass exchange, collision, interface promotion and interface creation
   advance the D3Q19 LBM–VOF state.
5. Smagorinsky–Lilly eddy viscosity augments the local relaxation time.
6. The completed fluid field and matching robot pose are copied to the renderer.
7. A ray-marched WebGPU view displays water and articulated geometry without CPU
   readback.

Simulation time advances only after submitted GPU work completes. Rendering frame
rate therefore cannot silently skip physical steps.

## Coupling model

The current robot is a prescribed kinematic boundary. Its linear and angular surface
velocities enter moving-wall bounce-back, transferring momentum to the fluid. Fluid
reaction forces are not integrated back into robot joints in this release.

## Boundary topology

The x and y boundaries are periodic. The z boundaries are stationary solid walls.
This keeps a compact local domain around the robot while allowing lateral waves to
continue across the cropped boundary.

## Precision

The production entry point uses FP32 storage and arithmetic. The solver source keeps
population-storage abstractions isolated so alternative storage formats can be
evaluated without changing the choreography or rendering layers.
