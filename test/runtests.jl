using SmolyakInterpolation, Test

"Simple function to test iterator."
function naive_capped_indices(cap, I)
    [Tuple(ι) for ι in CartesianIndices(map(i -> 1:i, I)) if sum(Tuple(ι)) ≤ cap]
end

@testset "capped cartesian indices" begin
    for _ in 1:100
        I = Tuple(rand(1:5, rand(1:5)))
        cap = sum(I) + rand(0:5)
        iter = CappedCartesianIndices(cap, I)
        collect(iter) == naive_capped_indices(cap, I)
    end
    @test_throws ArgumentError CappedCartesianIndices(1, (3, 3,)) # sum too low
    @test_throws ArgumentError CappedCartesianIndices(5, (0, 3,)) # negative
end
