module SmolyakInterpolation

export CappedCartesianIndices, gridpoint_set, set_length_counts, polynomials_at,
    degrees_of_freedom, basis_and_coordinates

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES
using Parameters: @unpack
using StaticArrays: SVector

include("utilities.jl")
include("univariate.jl")

####
#### capped cartesian indices iterator
####

struct CappedCartesianIndices{N}
    cap::Int
    I::NTuple{N,Int}
    function CappedCartesianIndices(cap::Int, I::NTuple{N,Int}) where {N}
        @argcheck all(I .≥ 1)
        @argcheck cap ≥ N
        new{N}(cap, I)
    end
end

Base.IteratorSize(::Type{<:CappedCartesianIndices}) = Base.HasLength()

function Base.length(iter::CappedCartesianIndices)
    @unpack cap, I = iter
    C = mapfoldr((C1, C2) -> convolve_counts(cap, C1, C2), I) do i
        i = min(cap, i)
        Counts(1:i, ones(Int, i))
    end
    sum(C.counts)
end

Base.IteratorEltype(::Type{<:CappedCartesianIndices}) = Base.HasEltype()

eltype(::Type{CappedCartesianIndices{N}}) where {N} = NTuple{N,Int}

@inline function Base.iterate(iter::CappedCartesianIndices{N}) where {N}
    ι = ntuple(_ -> 1, Val{N}())
    ι, ι
end

@inline function Base.iterate(iter::CappedCartesianIndices, ι)
    @unpack cap, I = iter
    valid, ι = __inc(cap, sum(ι), I, ι)
    valid ? (ι, ι) : nothing
end

@inline __inc(::Int, ::Int, ::Tuple{}, ::Tuple{}) = false, ()

@inline function __inc(cap::Int, ∑ι::Int, I::Tuple{Int}, ι::Tuple{Int})
    ι1 = first(ι)
    (ι1 < first(I) && ∑ι < cap), (ι1 + 1, )
end

@inline function __inc(cap, ∑ι, I, ι)
    ι1 = first(ι)
    if (ι1 < first(I) && ∑ι < cap)
        true, (ι1 + 1, Base.tail(ι)...)
    else
        Δ1 = 1 - ι1
        valid, ιtail = __inc(cap - 1, ∑ι + Δ1 - 1, Base.tail(I), Base.tail(ι))
        valid, (1, ιtail...)
    end
end

"""
$(SIGNATURES)

Largest individual index that will be in the values of the iterator.
"""
function largest_index(iter::CappedCartesianIndices{N}) where N
    @unpack cap, I = iter
    min(cap - N + 1, maximum(I))
end

function degrees_of_freedom(cap, I)
    C = mapfoldr((C1, C2) -> convolve_counts(cap, C1, C2), I) do i
        indexes = 1:min(i, cap)
        Counts(indexes, set_length.(indexes))
    end
    sum(C.counts)
end

prefix_combinations(x, y) = vec(((x, y) -> (x, y...)).(x, permutedims(y)))

"""
$(SIGNATURES)
"""
function basis_and_coordinates(cap::Int, I::NTuple{N,Int}) where {N}
    iter = CappedCartesianIndices(cap, I)
    K = largest_index(iter)
    R = set_range.(1:K)
    x0 = all_gridpoints(K)
    B0 = polynomials_at(last(last(R)), x0)
    Bij(i, j) = mapreduce(((i, j), ) -> B0[R[i], R[j]], kron, zip(i, j))
    xi(i) = mapfoldr(i -> x0[R[i]], prefix_combinations, i, init = [()])
    B = mapreduce(j -> mapreduce(i -> Bij(i, j), vcat, iter), hcat, iter)
    x = mapreduce(xi, vcat, iter)
    B, SVector.(x)
end

end # module
