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
include("Hyper-Evec-Centrality-master\\centrality.jl")
include("Hyper-Evec-Centrality-master\\data_io.jl")

using MatrixNetworks # Used for pagerank, graph data structure
using SparseArrays # Used for converting MatrixNetworks and matrices for PageRanking
using Profile # Used for performance measurement
using Statistics
using StatsBase
using PyCall

const mpr = pyimport("Enron.pagerank_motif")

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
    - `ϵ::Float64(=1.0e-6)`: Tolerance in calculating the PageRank vector

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
    - `ϵ::Float64(=1.0e-6)`: Tolerance in calculating the PageRank vector

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

"""
`n_argmax`
==========

Returns the indices of the top `n` maximal elements in descending order of array value.

Arguments
---------
    - `A::Vector{T}`: A vector of real numbers
    - `n::Int64`: How many of the top indices to return
"""
function n_argmax(A::Vector{T}, n::Int64) where T <: Real
    S = copy(A) # Copy of A that gets modified during iteration
    len = min(length(A),n) # How long to make the returned list
    inds = zeros(Int64,len)

    for i = 1:len
        max_ind = argmax(S) # Index of the ith largest value of A
        S[max_ind] = 0
        inds[i] = max_ind
    end

    return inds
end

root = "C:\\Users\\Elizabeth Turner\\Documents\\Josh\\School\\F1\\Network Science\\HO-HyperRank-matrix\\data\\" # Replace with your data directory

congress = root * "congress-bills"
dblp = root * "coauth-DBLP"
enron = root * "email-Enron"
ubuntu = root * "tags-ask-ubuntu"

T = read_data_unweighted(enron, 5)
H = read_arb(enron)
G = dyadic_projection(H)
C = spzeros(Float64,G.n,G.n)

for t in collect(triangles(G))
    C[t[1],t[2]] += 1
    C[t[2],t[1]] += 1
    C[t[1],t[3]] += 1
    C[t[3],t[1]] += 1
    C[t[2],t[3]] += 1
    C[t[3],t[2]] += 1
end

motifs = all_motifs(H)
#MC = motif_cooccurence(H,motifs,Int8(8))

println(n_argmax(CEC(T)[1],10))
println(n_argmax(HEC(T)[1],10))
println(n_argmax(ZEC(T)[1],10))

println(n_argmax(pagerank(G,0.85),10))
println(n_argmax(motif_pagerank(Matrix(sparse(G)), Matrix(C))))
println(n_argmax(hyper_rank(H, motifs, Int8(8)),10))
println()
#println(sort(1:148,by=x->sum(G.vals[G.rp[x]:G.rp[x+1]-1]),rev=true)[1:10])
#println(sort(1:148,by=x->sum(C.nzval[C.colptr[x]:C.colptr[x+1]-1]),rev=true)[1:10])
#println(sort(1:148,by=x->sum(MC.nzval[MC.colptr[x]:MC.colptr[x+1]-1]),rev=true)[1:10])

# Shared nodes between top triangles and top hyper motifs: None!
# Shared nodes between top degree and top hyper motifs: 114, 55, 47

# There are only 500 edges that take part in at least one instance of motif 8?!
# Out of 20k+ edges? Surprising since there are 13.7k instances of motif 8

# On the other hand, there are 1762 edges that take part in at least one triangle.
