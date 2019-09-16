@testset "set length and range" begin
    ks = 1:10
    rs = set_range.(Chebyshev(), ks)
    @test rs[1:4] == [1:1, 2:3, 4:5, 6:9]
    @test all(@. last(rs[1:(end - 1)]) + 1 == first(rs[2:end])) # contiguous
    @test length.(rs) == set_length.(Chebyshev(), ks)           # consistent with length
end

@testset "gridpoint sets" begin
    @test gridpoint_set(Chebyshev(), 1) == [0]
    @test gridpoint_set(Chebyshev(), 2) == [-1, 1]
    @test gridpoint_set(Chebyshev(), 3) ≈ [-√2, √2]./2
    @test issorted(gridpoint_set(Chebyshev(), 4))
    x = sort(mapreduce(k -> gridpoint_set(Chebyshev(), k), vcat, 1:5))
    @test allunique(x)
    @test length(x) == 2^4 + 1
    @test x ≈ cospi.((16:-1:0) ./ 16)
end

@testset "Chebyshev polynomials" begin
    J = 19
    x = rand(J) .* 2 .- 1
    N = 7
    Z = @inferred evaluate(N, Chebyshev(), x)
    @test size(Z) == (J, N)
    @test Z ≈ ((t, n) -> cos(n * t)).(acos.(x), (0:(N-1))')

    @test size(evaluate(0, Chebyshev(), x)) == (J, 0)
    @test evaluate(1, Chebyshev(), x) == ones(J, 1)
    @test evaluate(2, Chebyshev(), x) == hcat(ones(J, 1), x)
end
