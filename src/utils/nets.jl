####################################
# Utilities for adjacency matrices #
####################################

using SparseArrays
using ArgCheck

"""
    rowsums(A)

Compute row sums of matrix `A`.
"""
rowsums(A) = vec(sum(A; dims = 2))

"""
    isleft_substochastic(A)

Check if matrix `A` is left-substochastic.
"""
function isleft_substochastic(A::AbstractMatrix)
    all(zero(eltype(A)) .<= sum(A; dims = 1) .<= one(eltype(A)))
end
    
"""
    erdosrenyi(N, p, directed)

Generate random adjacency matrix from Erdos-Renyi random graph ensemble with `N` nodes and connection probability `p`. The network is directed by default.
"""
function erdosrenyi(N::Integer, p::Real, directed = true)
    @argcheck 0.0 <= p <= 1.0
    @argcheck 0 < N
    A = spzeros(N, N)
    for i = 1:N
        if directed
            for j = 1:N
                if i != j && rand() < p
                    A[i, j] = 1.0
                end
            end
        else
            for j = 1:(i-1)
                if rand() < p
                    A[i, j] = 1.0
                    A[j, i] = 1.0
                end
            end
        end
    end
    A
end

"""
    rescale(A, w)

Rescale matrix `A` such that column sums are equal to `w`.
"""
function rescale end

function rescale(A::AbstractMatrix{T}, w::T) where T
    rescale(A, fill(w, size(A, 2)))
end

function rescale(A::AbstractMatrix{T}, w::AbstractVector{T}) where T
    @argcheck size(A, 2) == length(w)
    B = similar(A)
    for j = 1:size(A, 2)
        s = sum(A[:, j])
        if s > 0
            B[:, j] = A[:, j] .* w[j] ./ s
        else
            B[:, j] = A[:, j]
        end
    end
    B
end
