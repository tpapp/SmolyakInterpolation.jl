module SmolyakInterpolation

export CappedCartesianIndices, gridpoint_set, set_length_counts, polynomials_at

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES
using Parameters: @unpack

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
    C = mapfoldr(i -> Counts(1:i, ones(Int, i)), (C1, C2) -> convolve_counts(cap, C1, C2), I)
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
    ι1 < first(I), (ι1 + 1, )
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
        indexes = 1:i
        Counts(indexes, set_length.(indexes))
    end
    mapreduce(((i, c), ) -> i * c, pairs(C))
end

function basis_matrix(cap, I)
    maximum
end

end # module
