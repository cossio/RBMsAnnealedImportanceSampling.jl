#=
Annealed importance sampling (AIS) to estimate the partition function (and hence
the log-likelihood).
A nice explanation of AIS in general can be found in Goodfellow et al Deep Learning book.
Salakhutdinov et al (10.1145/1390156.1390266, http://www.cs.utoronto.ca/~rsalakhu/papers/bm.pdf)
discusses AIS for RBMs specifically.

AIS tends to understimate the log of the partition function (in probability).
In contrast, Reverse AIS estimator (RAISE) can be used to obtain a stochastic upper bound.
See http://proceedings.mlr.press/v38/burda15.html.
Combining the two we can "sandwiches" the true value to have an idea if the Monte Carlo
chains have converged.

Addendum: I think Burda's paper has a typo. The correct expression for the weights In
reverse AIS (which I use here) can be found in Upadhya et al 2015, Equation 10
(https://link.springer.com/chapter/10.1007/978-3-319-26535-3_62).

Bonus: A discussion of estimating partition function in RBMs, comparing several algorithms:

https://www.sciencedirect.com/science/article/pii/S0004370219301948

For a variant or RAISE: https://arxiv.org/abs/1511.02543
=#

"""
    ais(rbm0, rbm1, v0, βs)

Provided `v0` is an equilibrated sample from `rbm0`, returns `F` such that `mean(exp.(F))` is
an unbiased estimator of `Z1/Z0`, the ratio of partition functions of `rbm1` and `rbm0`.

!!! tip Use [`logmeanexp`](@ref)
    `logmeanexp(F)`, using the function `logmeanexp`[@ref] provided in this package,
    tends to give a better approximation of `log(Z1) - log(Z0)` than `mean(F)`.
"""
function ais(rbm0::RBM, rbm1::RBM, v::AbstractArray, βs::AbstractVector)
    @assert issorted(βs) && 0 == first(βs) ≤ last(βs) == 1
    F0 = free_energy(rbm0, v) # β = 0 case
    F = zero(F0)
    for β in βs
        if 0 < β < 1
            rbm = anneal(rbm0, rbm1; β)
            F_prev = free_energy(rbm, v)
            v = sample_v_from_v(rbm, v)
            F_next = free_energy(rbm, v)
            F += F_next - F_prev
        end
    end
    F1 = free_energy(rbm1, v) # β = 1 case
    F += F0 - F1
    return F
end

function ais(rbm0::RBM, rbm1::RBM, v0::AbstractArray; nbetas::Int=2)
    βs = range(0, 1; length = nbetas)
    return ais(rbm0, rbm1, v0, βs)
end

"""
    aise(rbm, [βs]; [nbetas], init=rbm.visible, nsamples=1)

AIS estimator of the log-partition function of `rbm`. It is recommended to fit `init` to
the single-site statistics of `rbm` (or the data).

!!! tip Use large `nbetas`
    For more accurate estimates, use larger `nbetas`. It is usually better to have
    large `nbetas` and small `nsamples`, rather than large `nsamples` and small `nbetas`.
"""
function aise(rbm::RBM, βs::AbstractVector{<:Real}; init::AbstractLayer=rbm.visible, nsamples::Int=1)
    rbm0 = anneal_zero(init, rbm)
    v0 = sample_from_inputs(init, Falses(size(init)..., nsamples))
    F = ais(rbm0, rbm, v0, βs)
    return F .+ log_partition_zero_weight(rbm0)
end

"""
    raise(rbm::RBM, βs; v, init)

Reverse AIS estimator of the log-partition function of `rbm`.
While `aise` tends to understimate the log of the partition function, `raise` tends to
overestimate it. `v` must be an equilibrated sample from `rbm`.

!!! tip Use [`logmeanexp`](@ref)
    If `F = raise(...)`, then `-logmeanexp(-F)`, using the function `logmeanexp`[@ref]
    provided in this package, tends to give a better approximation of `log(Z)` than `mean(F)`.

!!! tip Sandwiching the log-partition function
    If `Rf = aise(...)`, `Rr = raise(...)` are the AIS and reverse AIS estimators, we have the
    stochastic bounds `logmeanexp(Rf) ≤ log(Z) ≤ -logmeanexp(-Rr)`.
"""
function raise(rbm::RBM, βs::AbstractVector; v::AbstractArray, init::AbstractLayer=rbm.visible)
    rbm0 = anneal_zero(init, rbm)
    F = ais(rbm, rbm0, v, βs)
    return log_partition_zero_weight(rbm0) .- F
end

aise(rbm::RBM; nbetas::Int=10000, kw...) = aise(rbm, range(0, 1, nbetas); kw...)
raise(rbm::RBM; nbetas::Int=10000, kw...) = raise(rbm, range(0, 1, nbetas); kw...)

"""
    log_partition_zero_weight(rbm)

Log-partition function of a zero-weight version of `rbm`.
"""
log_partition_zero_weight(rbm) = cgf(rbm.visible) + cgf(rbm.hidden)
