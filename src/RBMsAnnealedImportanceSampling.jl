module RBMsAnnealedImportanceSampling

using Base: front, tail
using FillArrays: Zeros, Falses
using LogStatFunctions: logmeanexp
using RestrictedBoltzmannMachines: RBM, AbstractLayer,
    Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU,
    cgf, free_energy, sample_v_from_v, sample_from_inputs

include("ais.jl")
include("anneal.jl")

end # module
