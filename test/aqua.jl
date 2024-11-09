import Aqua
import RBMsAnnealedImportanceSampling
using Test: @testset

@testset "aqua" begin
    Aqua.test_all(
        RBMsAnnealedImportanceSampling;
        stale_deps = (ignore = [:DiffRules],),
        ambiguities = (exclude = [reshape],)
    )
end
