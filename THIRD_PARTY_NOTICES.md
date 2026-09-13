# Third-Party Notices

## WebLBM

The TypeScript/WGSL module organization and separation of initialization, time
stepping and rendering were informed by
[LennartPaduch/WebLBM](https://github.com/LennartPaduch/WebLBM), inspected at commit
`427ed877da29faeabc8df47fb08863eec90441cb`.

WebLBM is distributed under the MIT License. A copy is included at
[`licenses/WebLBM-MIT.md`](licenses/WebLBM-MIT.md).

## FluidX3D

The free-surface formulation, shifted-population storage concepts and the
Smagorinsky–Lilly extension were informed by FluidX3D source distributions by
Moritz Lehmann. This repository contains a modified WGSL adaptation rather than the
original OpenCL implementation and is not an official FluidX3D release.

The applicable source-available licenses are included unchanged at:

- [`licenses/FluidX3D-1.0-LICENSE.md`](licenses/FluidX3D-1.0-LICENSE.md)
- [`licenses/FluidX3D-Intel-B70-LICENSE.md`](licenses/FluidX3D-Intel-B70-LICENSE.md)

These terms restrict commercial use, military use and, in the newer license,
training AI models on the source code. They also require attribution and publication
of modified source when binaries or generated results are published. Users are
responsible for complying with the applicable version and obtaining separate
permission from the copyright holder for uses outside those terms.

## Local CUDA reference implementation

The original numerical port was validated against a local CUDA LBM–VOF reference
case supplied by the project author. No CUDA source is included in this repository.
The public WebGPU implementation is an independently structured TypeScript/WGSL
adaptation.
