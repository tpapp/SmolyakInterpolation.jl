@testset "counts" begin
    for _ in 1:100
        a1, a2 = rand(1:10), rand(1:10)
        b1, b2 = a1 + rand(0:10), a2 + rand(0:10)
        c1 = rand(1:20, b1 - a1 + 1)
        c2 = rand(1:20, b2 - a2 + 1)
        cap = rand((a1+a2):(b1+b2))
        D = Dict{Int,Int}()
        for (i1, c1) in zip(a1:b1, c1)
            for (i2, c2) in zip(a2:b2, c2)
                k = i1 + i2
                if k ≤ cap
                    D[k] = get(D, k, 0) + c1 * c2
                end
            end
        end
        C = count_combinations(cap, Counts(a1:b1, c1), Counts(a2:b2, c2))
        for (i, c) in pairs(C)
            @test get(D, i, 0) == c
        end
        i1, i2 = extrema(keys(D))
        @test firstindex(C) ≤ i1
        @test i2 ≤ lastindex(C) ≤ cap
    end
end

"Simple function to test iterator."
function naive_capped_indices(cap, I)
    [Tuple(ι) for ι in CartesianIndices(map(i -> 1:i, I)) if sum(Tuple(ι)) ≤ cap]
end

@testset "capped cartesian indices" begin
    for _ in 1:100
        I = Tuple(rand(1:5, rand(1:5)))
        cap = length(I) + rand(0:5)
        iter = CappedCartesianIndices(cap, I)
        naive = naive_capped_indices(cap, I)
        @test length(naive) == @inferred length(iter)
        @test collect(iter) == naive
        @test @inferred(largest_indices(iter)) ==
            Tuple(maximum(mapreduce(x -> [x...], hcat, naive); dims = 2))
    end
    @test_throws ArgumentError CappedCartesianIndices(1, (3, 3,)) # sum too low
    @test_throws ArgumentError CappedCartesianIndices(5, (0, 3,)) # negative
end
