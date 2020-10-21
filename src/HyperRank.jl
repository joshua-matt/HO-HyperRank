"""
`HyperRank.jl`
============

This file contains implementations of [motif-based PageRank](http://www.cse.ust.hk/~yqsong/papers/2018-AAAI-MotifPageRank.pdf)
and the novel Higher-Order HyperRank. Higher-Order HyperRank is a generalization
of motif-based PageRank to hypergraphs, where hypergraph motifs are defined as in
https://arxiv.org/pdf/2003.01853.pdf.

The high-level idea is to perform a hypergraph PageRank in which the random walker
is more likely to traverse hyperedges that take part in more hypergraph motif
instances. However, in practice, the problem is equivalent to performing PageRank
on the projected graph of the hypergraph, where edges between vertices sharing
hyperedges that take part in hypergraph motifs are more likely to be traversed.
In other words, the probability that the random walker goes to a node with which
it shares many hypergraph motifs is increased. Thus, Higher-Order HyperRank can
be computed as a motif-based PageRank with a slightly different motif co-occurrence
matrix definition.
"""

include("Hypergraph.jl") # Hypergraph data structur
include("h_motifs.jl") # Hypergraph-motif finding, classification

using MatrixNetworks # Used for pagerank, graph data structure
using SparseArrays # Used for converting MatrixNetworks and matrices for PageRanking
using Profile # Used for performance measurement

"""
`motif_pagerank`
================

Perform [motif-based PageRank](http://www.cse.ust.hk/~yqsong/papers/2018-AAAI-MotifPageRank.pdf)
on a network.

Arguments
---------
    - `G::Matrix{T}`: The adjacency matrix of the network
    - `M::Matrix{T}`: The motif co-occurrence matrix of the network
    - `d::Float64(=0.5)`: The balance of the weighted average between `G` and `M`.
                          Must be in [0,1]. `d=1.0` means the PageRanked matrix
                          is G, and `d=0.0` means the PageRanked matrix is M.
    - `α::Float64(=0.85)`: The PageRank damping term. Must be in [0,1).
    - `ϵ::Float64(=1.0e-6)`: Tolerance in calulating the PageRank vector

Returns
-------
A vector `x`, where `x[i]` is the motif-based PageRank of node `i`
"""
function motif_pagerank(G::Matrix{T}, M::Matrix{V}; d::Float64=0.5, α::Float64=0.85, ϵ::Float64=1.0e-6) where T <: Real where V <: Real
    if d < 0 || d > 1 throw(ArgumentError("Expected d ∈ [0,1], got $d.")) end # Invalid d
    if α < 0 || α >= 1 throw(ArgumentError("Expected α ∈ [0,1), got $d.")) end # Invalid α

    average = d*(G/maximum(G)) + (1-d)*(M/maximum(M))
    return pagerank(MatrixNetwork(sparse(average)), α, ϵ) # PageRank the weighted average of G and M
end

"""
`hyper_rank`
============

Perform Higher-Order HyperRank on a hypergraph.

Arguments
---------
    - `M::MatrixHypergraph`: The hypergraph to analyze
    - `m::Int8`: The hypergraph motif id, must be in [1,26]. The motifs corresponding
                 to each id can be found on page 3 of the [paper](http://www.cse.ust.hk/~yqsong/papers/2018-AAAI-MotifPageRank.pdf).
    - `d::Float64(=0.5)`: The balance of the weighted average between `G` and `M`.
                          Must be in [0,1]. `d=1.0` means the PageRanked matrix
                          is G, and `d=0.0` means the PageRanked matrix is M.
    - `α::Float64(=0.85)`: The PageRank damping term. Must be in [0,1).
    - `ϵ::Float64(=1.0e-6)`: Tolerance in calculating the PageRank vector.

Returns
-------
A vector `x`, where `x[i]` is the Higher-Order HyperRank of node `i`
"""
function hyper_rank(M::MatrixHypergraph, m::Int8; d::Float64=0.5, α::Float64=0.85, ϵ::Float64=1.0e-6)
    if m < 1 || m > 26 throw(ArgumentError("Expected m ∈ [1,26], got $m.")) end

    G = dyadic_projection(M) # Projection of the hypergraph
    motifs = all_motifs(M) # Find and classify motif instances
    W = motif_cooccurence(M,motifs,m) # Construct the motif co-occurrence matrix
    return motif_pagerank(Matrix(sparse(G)),Matrix(W); d=d, α=α, ϵ=ϵ)
end

"""
`hyper_rank`
============

Perform Higher-Order HyperRank on a hypergraph.

Arguments
---------
    - `M::MatrixHypergraph`: The hypergraph to analyze
    - `motifs::Vector{Vector{Tuple{Int64,Int64,Int64}}}`: The categorization of all triples of hyperedges into hypergraph motifs,
                                                          where motifs[i] contains all triples that are instances of motif `i`
    - `m::Int8`: The hypergraph motif id, must be in [1,26]. The motifs corresponding
                 to each id can be found on page 3 of the [paper](http://www.cse.ust.hk/~yqsong/papers/2018-AAAI-MotifPageRank.pdf).
    - `d::Float64(=0.5)`: The balance of the weighted average between `G` and `M`.
                          Must be in [0,1]. `d=1.0` means the PageRanked matrix
                          is G, and `d=0.0` means the PageRanked matrix is M.
    - `α::Float64(=0.85)`: The PageRank damping term. Must be in [0,1).
    - `ϵ::Float64(=1.0e-6)`: Tolerance in calculating the PageRank vector.

Returns
-------
A vector `x`, where `x[i]` is the Higher-Order HyperRank of node `i`
"""
function hyper_rank(M::MatrixHypergraph, motifs::Vector{Vector{Tuple{Int64,Int64,Int64}}}, m::Int8; d::Float64=0.5, α::Float64=0.85, ϵ::Float64=1.0e-6)
    if m < 1 || m > 26 throw(ArgumentError("Expected m ∈ [1,26], got $m.")) end

    G = dyadic_projection(M) # Projection of the hypergraph
    W = motif_cooccurence(M,motifs,m) # Construct the motif co-occurrence matrix
    return motif_pagerank(Matrix(sparse(G)),Matrix(W); d=d, α=α, ϵ=ϵ)
end
