module SmolyakInterpolation

export CappedCartesianIndices, gridpoint_set, polynomials_at

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES
using Parameters: @unpack

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

Base.IteratorSize(::Type{<:CappedCartesianIndices}) = Base.SizeUnknown() # FIXME and calculate

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

####
#### disjoint sets of gridpoints
####

"""
$(SIGNATURES)

Set of additional (Chebyshev-Lobatto) gridpoints for index `k ≥ 1`. Sets are disjoint,
contain elements in increasing order, which for `k ∈ 2:K` contain the Chebyshev-Lobatto
gridpoints

```math
cos(πj / (N+ 1))
```
for ``j=0, …, N+1`` where ``N = 2ᴷ``, in an interleaved pattern. `k == 1` is just `[0.0]`.
"""
function gridpoint_set(k::Integer)
    if k == 1
        [0.0]
    elseif k == 2
        [-1.0, 1.0]
    else
        @argcheck k ≥ 1
        N = 2^(k - 1)
        cospi.(((N-1):-2:1) ./ N)
    end
end

####
#### Chebyshev polynomials
####

"""
$(SIGNATURES)

First `N` Chebyshev polynomials, evaluated at `x`, as columns of a matrix.
"""
function polynomials_at(N::Integer, x::AbstractVector{T}) where {T}
    @argcheck N ≥ 0
    Z = ones(T, length(x), N)
    if N ≥ 2
        Z[:, 2] = x
    end
    for j in 3:N
        @. Z[:, j] = 2 * x * Z[:, j - 1] - Z[:, j - 2]
    end
    Z
end

end # module
