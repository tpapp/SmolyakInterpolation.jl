#####
##### univariate building blocks
#####

"""
$(SIGNATURES)

Represents the sequence of Chebyshev polynomials (of the first kind).
"""
struct Chebyshev end

Base.Broadcast.broadcastable(C::Chebyshev) = Ref(C)

####
#### set length calculations
####

"""
$(SIGNATURES)

Length of a disjoint set with index `k ≥ 1`.
"""
function set_length(::Chebyshev, k::Integer)
    @argcheck k > 0
    if k == 1
        1
    elseif k == 2
        2
    else
        2^(k - 2)
    end
end

"""
$(SIGNATURES)

Indexes for disjoint set with index `k ≥ 1`.
"""
function set_range(::Chebyshev, k::Integer)
    @argcheck k > 0
    if k == 1
        1:1
    elseif k == 2
        2:3
    else
        K = 2^(k - 2)
        start = 2 + K
        start:(start + K - 1)
    end
end

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
function gridpoint_set(::Chebyshev, k::Integer)
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

function all_gridpoints(::Chebyshev, K::Integer)
    mapreduce(k -> gridpoint_set(Chebyshev(), k), vcat, 1:K)
end

# struct ChebyshevPolynomials{T}
#     x::T
# end

# Base.IteratorEltype(::Type{ChebyshevPolynomials{T}}) where {T} = float(T)

# Base.IteratorSize(::Type{<:ChebyshevPolynomials}) = Base.IsInfinite()

# function Base.iterate(c::ChebyshevPolynomials{T}) where {T}
#     S = float(T)
#     one(S), (one(S) / 2, zero(S))
# end

# function Base.iterate(c::ChebyshevPolynomials, (yp, ypp))
#     y = 2 * c.x * yp - ypp
#     y, (y, yp)
# end

function evaluate!(z::AbstractVector, ::Chebyshev, x::Real)
    N = length(z)
    if N ≥ 1
        ypp = z[1] = 1
        if N ≥ 2
            yp = z[2] = x
            for i in 3:N
                y = 2 * yp * x - ypp
                ypp = yp
                yp = y
                z[i] = y
            end
        end
    end
    z
end

"""
$(SIGNATURES)

Evaluate the first `N` basis functions at `x`, returning a vector.
"""
function evaluate(N::Integer, C::Chebyshev, x::Real)
    evaluate!(Vector{float(eltype(x))}(undef, length(x)), C, x)
end

"""
$(SIGNATURES)

In-place version of [`evaluate`](@ref), returns the first argument.
"""
function evaluate!(Z::AbstractMatrix, ::Chebyshev, x::AbstractVector)
    N = size(Z, 2)
    @argcheck size(Z, 1) == length(x)
    if N ≥ 1
        Z[:, 1] .= 1
        if N ≥ 2
            Z[:, 2] .= x
            for j in 3:N
                @. Z[:, j] = 2 * x * Z[:, j - 1] - Z[:, j - 2]
            end
        end
    end
    Z
end

"""
$(SIGNATURES)

Evaluate the first `N` basis functions at elements of `x`, one row for each element.
"""
function evaluate(N::Integer, C::Chebyshev, x::AbstractVector)
    evaluate!(Matrix{float(eltype(x))}(undef, length(x), N), C, x)
end
