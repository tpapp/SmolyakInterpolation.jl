###
### length calculations
###

struct Counts
    indexes::UnitRange{Int}
    counts::Vector{Int}
end

Base.pairs(C::Counts) = zip(C.indexes, C.counts)

function convolve_counts(cap, C1::Counts, C2::Counts)
    i1, c1 = C1.indexes, C1.counts
    i2, c2 = C2.indexes, C2.counts
    i_first = first(i1) + first(i2)
    i_last = min(cap, last(i1) + last(i2))
    c = zeros(Int, i_last - i_first + 1)
    for (i1, c1) in pairs(C1)
        for (i2, c2) in pairs(C2)
            i = i1 + i2
            if i ≤ i_last
                c[i - i_first + 1] += c1 * c2
            end
        end
    end
    Counts(i_first:i_last, c)
end
