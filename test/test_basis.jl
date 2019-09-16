@testset "basis and coordinates" begin
    B = HomogeneousBasis(Chebyshev(), (4, 4), 1)
    A, x = basis_matrix_and_coordinates(B)
    d = degrees_of_freedom(B)
    @test size(A) == (d, d)
    @test size(x) == (d, )
    @test eltype(x) == SVector{2, Float64}
end
