# Environment Roadmap

AquaEmbodied WebGPU treats the surrounding medium as part of embodied intelligence.
The architecture is intended to support robots in environments where air-only
assumptions fail.

## Implemented: free-surface water

The current release resolves a robot disturbing a three-dimensional water surface.
It supports gravity, viscosity, surface tension, contact angle, water depth and
moving solid boundaries.

## Submerged and deep-ocean-like scenarios

The existing lattice field and moving-boundary representation can be reused for a
fully filled domain. A scientifically faithful deep-ocean case additionally needs:

- pressure-consistent initialization at depth;
- an equation of state or appropriate incompressible pressure treatment;
- pressure-rated robot dynamics and buoyancy feedback;
- validation against subsea benchmark data.

Viscosity and gravity sweeps in the current code can explore qualitative non-air
media, but should not be presented as validated deep-ocean pressure simulations.

## Reduced gravity and space-fluid scenarios

Configurable gravity and closed boundaries provide a starting point for contained
liquid motion under reduced gravity. A complete space-robotics model additionally
needs six-degree-of-freedom spacecraft dynamics, wetting validation, capillary-scale
resolution and conservation-aware two-way coupling. Vacuum itself is not a fluid and
is outside the LBM domain.

## Physical AI integration

Planned interfaces will expose observations, actions and physically meaningful
metrics for controller evaluation and reinforcement learning. Candidate tasks
include splash minimization, submerged gait adaptation, fluid-aware manipulation,
microgravity liquid management and robust motion planning across media.
