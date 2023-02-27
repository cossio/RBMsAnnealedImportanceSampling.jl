"""
    anneal(rbm0, rbm1; β)

Returns an RBM that interpolates between `rbm0` and `rbm1`.
Denoting by `E0(v, h)` and `E1(v, h)` the energies assigned by `rbm0` and `rbm1`,
respectively, the returned RBM assigns energies given by:

    E(v,h) = (1 - β) * E0(v) + β * E1(v, h)
"""
function anneal(rbm0::RBM, rbm1::RBM; β::Real)
    vis = anneal(rbm0.visible, rbm1.visible; β)
    hid = anneal(rbm0.hidden, rbm1.hidden; β)
    w = (1 - β) * rbm0.w + β * rbm1.w
    return RBM(vis, hid, w)
end

function anneal(init::Binary, final::Binary; β::Real)
    θ = (1 - β) * init.θ + β * final.θ
    return Binary(; θ)
end

function anneal(init::Spin, final::Spin; β::Real)
    θ = (1 - β) * init.θ + β * final.θ
    return Spin(; θ)
end

function anneal(init::Potts, final::Potts; β::Real)
    θ = (1 - β) * init.θ + β * final.θ
    return Potts(; θ)
end

function anneal(init::Gaussian, final::Gaussian; β::Real)
    θ = (1 - β) * init.θ + β * final.θ
    γ = (1 - β) * init.γ + β * final.γ
    return Gaussian(; θ, γ)
end

function anneal(init::ReLU, final::ReLU; β::Real)
    θ = (1 - β) * init.θ + β * final.θ
    γ = (1 - β) * init.γ + β * final.γ
    return ReLU(; θ, γ)
end

function anneal(init::dReLU, final::dReLU; β::Real)
    θp = (1 - β) * init.θp + β * final.θp
    θn = (1 - β) * init.θn + β * final.θn
    γp = (1 - β) * init.γp + β * final.γp
    γn = (1 - β) * init.γn + β * final.γn
    return dReLU(; θp, θn, γp, γn)
end

function anneal(init::pReLU, final::pReLU; β::Real)
    θ = (1 - β) * init.θ + β * final.θ
    γ = (1 - β) * init.γ + β * final.γ
    Δ = (1 - β) * init.Δ + β * final.Δ
    η = (1 - β) * init.η + β * final.η
    return pReLU(; θ, γ, Δ, η)
end

function anneal(init::xReLU, final::xReLU; β::Real)
    θ = (1 - β) * init.θ + β * final.θ
    γ = (1 - β) * init.γ + β * final.γ
    Δ = (1 - β) * init.Δ + β * final.Δ
    ξ = (1 - β) * init.ξ + β * final.ξ
    return xReLU(; θ, γ, Δ, ξ)
end

anneal_zero(init::AbstractLayer, rbm::RBM) = RBM(init, anneal_zero(rbm.hidden), Zeros(rbm.w))

anneal_zero(l::Binary) = Binary(; θ = zero(l.θ))
anneal_zero(l::Spin) = Spin(; θ = zero(l.θ))
anneal_zero(l::Potts) = Potts(; θ = zero(l.θ))
anneal_zero(l::Gaussian) = Gaussian(; θ = zero(l.θ), l.γ)
anneal_zero(l::ReLU) = ReLU(; θ = zero(l.θ), l.γ)
anneal_zero(l::dReLU) = dReLU(; θp = zero(l.θp), θn = zero(l.θn), l.γp, l.γn)
anneal_zero(l::pReLU) = pReLU(; θ = zero(l.θ), l.γ, Δ = zero(l.Δ), l.η)
anneal_zero(l::xReLU) = xReLU(; θ = zero(l.θ), l.γ, Δ = zero(l.Δ), l.ξ)
