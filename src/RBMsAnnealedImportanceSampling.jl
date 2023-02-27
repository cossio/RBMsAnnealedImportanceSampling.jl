module RBMsAnnealedImportanceSampling

using Base: front, tail
using LogExpFunctions: logsumexp, logsubexp
using FillArrays: Zeros, Falses
using RestrictedBoltzmannMachines: RBM, AbstractLayer,
    Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU,
    cgf, free_energy, sample_v_from_v, sample_from_inputs

include("ais.jl")
include("anneal.jl")
include("util.jl")

end # module
