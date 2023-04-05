# RBMsAnnealedImportanceSampling Julia package

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/cossio/RBMsAnnealedImportanceSampling.jl/blob/master/LICENSE.md)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://cossio.github.io/RBMsAnnealedImportanceSampling.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://cossio.github.io/RBMsAnnealedImportanceSampling.jl/dev)
![](https://github.com/cossio/RBMsAnnealedImportanceSampling.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/cossio/RBMsAnnealedImportanceSampling.jl/branch/master/graph/badge.svg?token=O5P8LQTVF3)](https://codecov.io/gh/cossio/RBMsAnnealedImportanceSampling.jl)
![GitHub repo size](https://img.shields.io/github/repo-size/cossio/RBMsAnnealedImportanceSampling.jl)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/cossio/RBMsAnnealedImportanceSampling.jl)

Annealed importance sampling algorithm to estimate the partition function of Restricted Boltzmann machines. The package is registered. Install it with:

```julia
import Pkg
Pkg.add("RBMsAnnealedImportanceSampling")
```

## Related packages

Restricted Boltzmann machines in Julia:

- https://github.com/cossio/RestrictedBoltzmannMachines.jl

Use RBMs on the GPU (CUDA):

- https://github.com/cossio/CudaRBMs.jl

Centered RBMs:

- https://github.com/cossio/CenteredRBMs.jl

## Citation

If you use this package in a publication, please cite:

* Jorge Fernandez-de-Cossio-Diaz, Simona Cocco, and Remi Monasson. "Disentangling representations in Restricted Boltzmann Machines without adversaries." Physical Review X 13, 021003 (2023).

Or you can use the included [CITATION.bib](https://github.com/cossio/RestrictedBoltzmannMachines.jl/blob/master/CITATION.bib).