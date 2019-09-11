#####
##### univariate building blocks
#####


####
#### set length calculations
####

"""
$(SIGNATURES)

Length of a disjoint set with index `k ≥ 1`.
"""
function set_length(k::Integer)
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
function set_range(k::Integer)
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
