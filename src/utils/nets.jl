####################################
# Utilities for adjacency matrices #
####################################

using SparseArrays
using ArgCheck
import StatsBase

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

Generate random adjacency matrix from Erdos-Renyi random graph
ensemble with `N` nodes and connection probability `p`. The network
is directed by default.
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

"""
    barabasialbert(N, m, initgraph)

Creates a random graph based on preferential attachment, as described by
the model by Barabasi and Albert. The graph has 'N' nodes each with 'm'
edges. 'initgraph' is a name of the method to initialize the
subgraph which new nodes will be added onto.
"""
function barabasialbert(N::Integer, m::Integer, initgraph="random")
    @argcheck 1 < m < N

    # create initial random subgraph
    if initgraph == "random"
        A = initm0graph(N, m)
    end

    # create array in which each node occurs as many times as it has edges
    inds = findall(x -> x != 0.0, A)
    repeatednodes = [inds[i][2] for i=1:size(inds)[1]]

    newnode = m+1
    # preferential attachment
    while newnode <= N
        # sample 'm' new neighbour nodes based on their degree weights
        j = StatsBase.sample(1:newnode-1,
                    StatsBase.Weights(attachmentweights(repeatednodes)),
                    m, replace=false)

        A[newnode, j] .= 1.0
        A[j, newnode] .= 1.0

        # add new edges to list of repeated nodes
        append!(repeatednodes, j)
        append!(repeatednodes, newnode*ones(size(j)[1]))

        newnode += 1
    end
    return A
end

"""
    initm0graph(N, m0)

Creates initial random graph of 'm0' connected nodes. 'm0' < 'N' and
the resulting random graph is a subgraph of the entire graph specified
by the adjacency matrix 'A'. This subgraph is required for the
Barabasi-Albert model.
"""
function initm0graph(N::Integer, m0::Integer)
    @argcheck 1 < m0 < N
    # adjacency matrix of entire graph
    A = spzeros(N, N)
    # list of all nodes of subgraph
    nodes = [1:m0;]
    # create initial, randomly connected graph
    for i in nodes
        # from the list of nodes randomly sample x nodes that the current
        # node is connected to, whereby x is also randomly sampled
        j = StatsBase.sample(nodes[1:end .!= i], rand(1:m0-1), replace = false)
        A[i, j] .= 1.0
        A[j, i] .= 1.0
    end
    return A
end

"""
    attachmentweights(repeatednodes)

Compute weights for preferential attachment. For each node of the graph
specified in the array of repeated nodes, 'repeatednodes', compute
``k\\_i/(\\sum\\_j k\\_j)`` where ``k\\_i`` is the degree of node i.
Specifically, 'repeatednodes' is an array of nodes, where each node
occurs as many times as it has edges.
"""
function attachmentweights(repeatednodes::Array{Int64})
    nodes = unique(repeatednodes)
    # array to store the weights
    w = ones(length(nodes))
    for i in nodes
        # degree of node divided by total number of edges
        w[i] = length(findall(x -> x == i, repeatednodes))/
                length(repeatednodes)
    end
    # end
    return w
end
