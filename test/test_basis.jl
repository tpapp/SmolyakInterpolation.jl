@testset "building blocks" begin
    A = rand(5, 5)
    x = rand(5)
    R = [1:2, 3:5]
    @test _basis_block(A, R, (1, 2), (2, 1)) ==
        kron(A[R[1], R[2]], A[R[2], R[1]])
    @test @inferred(_basis_block(A, R, (1, 2, 2), (2, 1, 1))) ==
        kron(kron(A[R[1], R[2]], A[R[2], R[1]]), A[R[2], R[1]])
    @test @inferred(_coordinate_block(x, R, (1, 2, 2))) ==
        tuple.(repeat(x[R[1]], outer = 9),
               repeat(x[R[2]], inner = 2, outer = 3),
               repeat(x[R[2]], inner = 6))
end

@testset "basis and coordinates" begin
    B = HomogeneousBasis(Chebyshev(), (4, 4), 1)
    A, v = basis_matrix_and_coordinates(B)

    # consistency checks
    d = degrees_of_freedom(B)
    @test size(A) == (d, d)
    @test size(v) == (d, )
    @test eltype(v) == SVector{2, Float64}

    # interpolation
    f(x) = sin(x[1]) + cos(x[2] * 1.7)
    fx = f.(v)
    b = A \ fx

    # test at exactly interpolated points
    for i in 1:d
        @test interpolate(B, b, v[i]) ≈ fx[i]
    end
end
