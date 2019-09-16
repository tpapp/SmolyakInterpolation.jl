"""
$(TYPEDEF)

# Fields

Accessing fields directly is part of the API. The additional property `cap` is supported.

$(FIELDS)
"""
struct HomogeneousBasis{N,U}
    "The univariate family, eg `Chebyshev()`."
    univariate::U
    "Indexes by dimensions."
    I::NTuple{N,Int}
    "The level (`≥ 0`) of the Smolyak approximation. `cap = level + N`."
    level::Int
end

function Base.propertynames(basis::HomogeneousBasis, private = false)
    (fieldnames(HomogeneousBasis)..., :cap)
end

function Base.getproperty(basis::HomogeneousBasis{N}, name::Symbol) where {N}
    if name ≡ :cap
        N + basis.level
    else
        getfield(basis, name)
    end
end

function degrees_of_freedom(basis::HomogeneousBasis)
    @unpack univariate, I, cap = basis
    C = mapfoldl((C1, C2) -> count_combinations(cap, C1, C2), I) do i
        indexes = 1:min(i, cap)
        Counts(indexes, set_length.(univariate, indexes))
    end
    sum(C)
end

"""
$(SIGNATURES)

Return `A, v` where `A` is a ``d × d`` matrix and `v` is a `d`-element vector, where

- `d = degrees_of_freedom(basis)`,

- `v` is a vector of *coordinates*,

- `A \\ f.(x)` would calculate the *coefficients* of approximating `f`.
"""
function basis_matrix_and_coordinates(basis::HomogeneousBasis)
    @unpack univariate, I, cap = basis
    iter = CappedCartesianIndices(cap, I)
    K = largest_index(iter)

    # univariate building blocks
    R = set_range.(univariate, 1:K)
    x0 = all_gridpoints(univariate, K)
    A0 = evaluate(last(last(R)), univariate, x0)

    # combinations
    Aij(i, j) = mapreduce(((i, j), ) -> A0[R[i], R[j]], kron, zip(i, j))
    xi(i) = mapfoldl(i -> x0[R[i]],
                     (y, x) -> vec(((x, y) -> (x, y...)).(x, permutedims(y))),
                     i, init = [()])
    # FIXME these could be made much more efficient by pre-allocating
    A = mapreduce(j -> mapreduce(i -> Aij(i, j), vcat, iter), hcat, iter)
    x = mapreduce(xi, vcat, iter)
    A, SVector.(x)
end
