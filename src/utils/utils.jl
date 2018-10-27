"""
    rowsums(A)

Compute row sums of matrix `A`.
"""
rowsums(A) = vec(sum(A; dims = 2))

"""
    constantly(val)

Create constant function that returns `val` when called.
"""
constantly(val) = (x...) -> val

"""
    fillrows(v)

Fills a square matrix by repeating a vector `v` rowwise.
"""
function fillrows(v::AbstractVector)
    D = length(v)
    repeat(reshape(v, 1, D), D, 1)
end

"""
    isleft_substochastic(A)

Check if matrix `A` is left-substochastic.
"""
function isleft_substochastic(A::AbstractMatrix)
    all(zero(eltype(A)) .<= sum(A; dims = 1) .<= one(eltype(A)))
end
    
