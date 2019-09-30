####
#### length calculations
####

"""
$(TYPEDEF)

Counts for contiguous indexes. Used for combinatoric calculations.
"""
struct Counts
    indexes::UnitRange{Int}
    counts::Vector{Int}
end

Base.pairs(C::Counts) = zip(C.indexes, C.counts)

Base.firstindex(C::Counts) = first(C.indexes)

Base.lastindex(C::Counts) = last(C.indexes)

Base.sum(C::Counts) = sum(C.counts)

"""
$(SIGNATURES)

Calculate the counts
```math
C[k] = ∑ C₁[i] ⋅ C₂[j]  \text{where} i + j = k ≤ cap
```
"""
function count_combinations(cap, C1::Counts, C2::Counts)
    i_first = firstindex(C1) + firstindex(C2)
    i_last = min(cap, lastindex(C1) + lastindex(C2))
    c = zeros(Int, i_last - i_first + 1)
    for (i1, c1) in pairs(C1)
        for (i2, c2) in pairs(C2)
            i = i1 + i2
            if i ≤ i_last
                c[i - i_first + 1] += c1 * c2
            else
                break           # no point in continuing inner loop
            end
        end
    end
    Counts(i_first:i_last, c)
end

####
#### capped cartesian indices iterator
####

"""
$(TYPEDEF)

An iterator over the cartesian indexes `1 ≤ i[1] ≤ I[1]`, …, in colum major order, with
`sum(i) ≤ cap`. Iterator returns N-tuples of integers. Useful for traversing Smolyak
combinations.
"""
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
    C = mapfoldl((C1, C2) -> count_combinations(cap, C1, C2), I) do i
        i = min(cap, i)
        Counts(1:i, ones(Int, i))
    end
    sum(C)
end

Base.IteratorEltype(::Type{<:CappedCartesianIndices}) = Base.HasEltype()

Base.eltype(::Type{CappedCartesianIndices{N}}) where {N} = NTuple{N,Int}

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

Largest indices for each axis that occur in the iteration.
"""
function largest_indices(iter::CappedCartesianIndices{N}) where N
    @unpack cap, I = iter
    min.(cap - N + 1, I)
end
