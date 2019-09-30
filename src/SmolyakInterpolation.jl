module SmolyakInterpolation

export
    # chebyshev
    Chebyshev, set_length, set_range, gridpoint_set, evaluate, evaluate!,
    # basis
    HomogeneousBasis, degrees_of_freedom, basis_matrix_and_coordinates, interpolate,
    interpolated_basis!, interpolated_basis

using ArgCheck: @argcheck
using DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF
using Parameters: @unpack
using StaticArrays: SVector

include("utilities.jl")
include("chebyshev.jl")
include("basis.jl")

end # module
