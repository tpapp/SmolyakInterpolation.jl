@testset "building blocks" begin
    A = rand(5, 5)
    x = rand(5)
    R = [1:2, 3:5]
    @test _basis_block(A, R, (1, 2), (2, 1)) ==
        kron(A[R[2], R[1]], A[R[1], R[2]])
    @test @inferred(_basis_block(A, R, (1, 2, 2), (2, 1, 1))) ≈
        kron(kron(A[R[2], R[1]],
                  A[R[2], R[1]]),
             A[R[1], R[2]])
    @test @inferred(_coordinate_block(x, R, (1, 2, 2))) ==
        tuple.(repeat(x[R[1]], outer = 9),
               repeat(x[R[2]], inner = 2, outer = 3),
               repeat(x[R[2]], inner = 6))
end

@testset "basis and coordinates" begin
    B = HomogeneousBasis(Chebyshev(), (4, 4), 2)
    A, v = basis_matrix_and_coordinates(B)

    # consistency checks
    d = degrees_of_freedom(B)
    @test size(A) == (d, d)
    @test size(v) == (d, )
    @test eltype(v) == SVector{2, Float64}

    # compare to manually constructed matrix
    R = set_range.(Chebyshev(), 1:3)
    x0 = all_gridpoints(Chebyshev(), 4)
    A0 = evaluate(9, Chebyshev(), x0)

    iter = CappedCartesianIndices(4, (4, 4))
    let row_sum = 0             # cumulative row indx
        for i_row in iter
            col_sum = 0         # cumulative column index
            row_s = 0           # when ≠0, (row) size of blocks
            for i_col in iter
                a = fill(1.0, 1, 1)
                for (i, j) in zip(i_row, i_col)
                    a = kron(A0[R[i], R[j]], a)
                end
                s1, s2 = size(a)
                if row_s == 0
                    row_s = s1      # first, just save
                else
                    @test row_s == s1
                end
                @test a ≈ A[row_sum .+ (1:s1), col_sum .+ (1:s2)]
                col_sum += s2
            end
            row_sum += row_s
            @test col_sum == d
        end
        @test row_sum == d
    end
    let p = 0
        for i in iter
            for j in CartesianIndices(map(i -> R[i], i))
                p += 1
                @test v[p] ≈ SVector(map(j -> x0[j], Tuple(j)))
            end
        end
        @test p == d
    end

    # interpolation
    f(x) = sin(x[1]) + cos(x[2] * 1.7)
    fx = f.(v)
    b = A \ fx

    # test at exactly interpolated points
    for i in 1:d
        @test interpolate(B, b, v[i]) ≈ fx[i]
    end

    # test at other points
    xs = [rand(2) .* 2 .- 1 for _ in 1:50]
    for x in xs
        @test interpolate(B, b, x) ≈ f(x) atol = 1e-2
    end
end
