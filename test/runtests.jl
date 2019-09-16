using SmolyakInterpolation, Test, StaticArrays

# internals to be tested
using SmolyakInterpolation:
    # utilities
    Counts, count_combinations, CappedCartesianIndices, largest_index

include("test_utilities.jl")
include("test_chebyshev.jl")
include("test_basis.jl")
