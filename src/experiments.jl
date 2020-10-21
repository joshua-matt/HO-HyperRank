include("HyperRank.jl")
include("Hyper-Evec-Centrality-master\\centrality.jl")
include("Hyper-Evec-Centrality-master\\data_io.jl")

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

"""
`n_argmax`
==========

Returns the labels of the top `n` maximal elements in descending order of array value.

Arguments
---------
    - `A::Vector{T}`: A vector of real numbers
    - `n::Int64`: How many of the top indices to return
    - `labels`: Labels associated with indices of A
"""
function n_argmax(A::Vector{T}, n::Int64, labels) where T <: Real
    return labels[n_argmax(A,n)]
end


root = string(@__DIR__) * "\\..\\data\\" # This file must have the same parent directory as data folder

# Included datasets
congress = "congress-bills"
dblp = "coauth-DBLP"
enron = "email-Enron"
ubuntu = "tags-ask-ubuntu"

name = enron
file = root * name # Directory of selected data
topK = 10 # Print the top K vertices
use_labels = true # Whether to print labels of vertices or their numbers
label_replace = ("@enron.com","") # Allows replacement of string 1 with string 2 in reading the labels
T = read_data_unweighted(file, 5) # Hypergraph as tensor
H = read_arb(file) # Hypergraph as incidence matrix
G = dyadic_projection(H) # Projected graph
C = spzeros(Float64,G.n,G.n) # Triangle co-occurrence matrix
motifs = all_motifs(H) # Hypergraph motifs

for t in collect(triangles(G)) # Filling triangle co-occurrence matrix
    C[t[1],t[2]] += 1
    C[t[2],t[1]] += 1
    C[t[1],t[3]] += 1
    C[t[3],t[1]] += 1
    C[t[2],t[3]] += 1
    C[t[3],t[2]] += 1
end

if use_labels
    labels = replace.((x->x[2]).(split.(readlines(file * "\\" * name * "-node-labels.txt"))), label_replace[1] => label_replace[2])

    println("Clique Motif Eigenvector Centrality\n===================================")
    println(n_argmax(CEC(T)[1],topK,labels))
    println()
    println("Z-Eigenvector Centrality\n========================")
    println(n_argmax(ZEC(T)[1],topK,labels))
    println()
    println("H-Eigenvector Centrality\n========================")
    println(n_argmax(HEC(T)[1],topK,labels))
    println()
    println("Hypergraph PageRank\n===================")
    println(n_argmax(pagerank(G,0.85),topK,labels))
    println()
    println("Motif-based PageRank\n====================")
    println(n_argmax(motif_pagerank(Matrix(sparse(G)), Matrix(C)),topK,labels))
    println()
    println("Higher-Order HyperRank\n======================")
    println(n_argmax(hyper_rank(H, Int8(8)),topK,labels))
else
    println("Clique Motif Eigenvector Centrality\n===================================")
    println(n_argmax(CEC(T)[1],topK))
    println()
    println("Z-Eigenvector Centrality\n========================")
    println(n_argmax(ZEC(T)[1],topK))
    println()
    println("H-Eigenvector Centrality\n========================")
    println(n_argmax(HEC(T)[1],topK))
    println()
    println("Hypergraph PageRank\n===================")
    println(n_argmax(pagerank(G,0.85),topK))
    println()
    println("Motif-based PageRank\n====================")
    println(n_argmax(motif_pagerank(Matrix(sparse(G)), Matrix(C)),topK))
    println()
    println("Higher-Order HyperRank\n======================")
    println(n_argmax(hyper_rank(H, Int8(8)),topK))
end

  
