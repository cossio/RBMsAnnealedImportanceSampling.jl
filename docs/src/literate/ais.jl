#=
# Annealed importance sampling

We can compute the partition function of the RBM (and hence the log-likelihood) with
annealed importance sampling (AIS).
=#

import MLDatasets
import Makie
import CairoMakie
using Statistics: mean, std, middle
using RBMsAnnealedImportanceSampling: aise, raise, logmeanexp
using RestrictedBoltzmannMachines: Binary, BinaryRBM, initialize!, pcd!, sample_v_from_v

# Load MNIST (0 digit only).

train_x = MLDatasets.MNIST(split=:train)[:].features .> 0.5
train_y = MLDatasets.MNIST(split=:train)[:].targets
train_x = train_x[:, :, train_y .== 0]
nothing #hide

# Train an RBM

rbm = BinaryRBM((28,28), 128)
initialize!(rbm, train_x)
@time pcd!(rbm, train_x; iters=10000, batchsize=128)
nothing #hide

# Get some equilibrated samples from model
v = train_x[:, :, rand(1:size(train_x, 3), 1000)]
v = sample_v_from_v(rbm, v; steps=1000)
nothing #hide

# Estimate Z with AIS and reverse AIS.

nsamples=100
ndists = [10, 100, 1000, 10_000, 100_000]
R_ais = Vector{Float64}[]
R_rev = Vector{Float64}[]
init = initialize!(Binary(; θ = zero(rbm.visible.θ)), v)
nothing #hide

for nbetas in ndists
    push!(R_ais,
        @time aise(rbm; nbetas, nsamples, init)
    )
    push!(R_rev,
        @time raise(rbm; nbetas, init, v=v[:,:,rand(1:size(v, 3), nsamples)])
    )
end

# Plots

fig = Makie.Figure()
ax = Makie.Axis(
    fig[1,1], width=700, height=400, xscale=log10, xlabel="interpolating distributions", ylabel="log(Z)"
)
Makie.band!(
    ax, ndists,
    mean.(R_ais) - std.(R_ais),
    mean.(R_ais) + std.(R_ais);
    color=(:blue, 0.25)
)
Makie.band!(
    ax, ndists,
    mean.(R_rev) - std.(R_rev),
    mean.(R_rev) + std.(R_rev);
    color=(:black, 0.25)
)
Makie.lines!(ax, ndists, mean.(R_ais); color=:blue, label="AIS")
Makie.lines!(ax, ndists, mean.(R_rev); color=:black, label="reverse AIS")
Makie.lines!(ax, ndists, logmeanexp.(R_ais); color=:blue, linestyle=:dash)
Makie.lines!(ax, ndists, logmeanexp.(R_rev); color=:black, linestyle=:dash)
Makie.lines!(ax, ndists, -logmeanexp.(-R_rev); color=:orange, linestyle=:dash)
Makie.hlines!(ax, middle(mean(R_ais[end]), mean(R_rev[end])), linestyle=:dash, color=:red, label="limiting estimate")
Makie.xlims!(extrema(ndists)...)
Makie.axislegend(ax, position=:rb)
Makie.resize_to_layout!(fig)
fig
