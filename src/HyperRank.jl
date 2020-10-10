include("Hypergraph.jl")
include("h_motifs.jl")
using SparseArrays, LinearAlgebra
using MatrixNetworks
using Profile

function motif_pagerank(G::Matrix, M::Matrix; d=0.5, α=0.85, ϵ=1.0e-6) # TODO: Find way to add sparse matrices
    return pagerank(MatrixNetwork(sparse((d*G + (1-d)*M))), α, ϵ)
end

"""
`hyper_rank`
============
"""
function hyper_rank(M::MatrixHypergraph, m::Int8; d=0.5, α=0.85, ϵ=1.0e-6)
    G = dyadic_projection(M)
    motifs = all_motifs(M)
    W = motif_cooccurence(M,motifs,m)
    return motif_pagerank(Matrix(sparse(G)),Matrix(W); d=d, α=α, ϵ=ϵ)
end

function hyper_rank(M::MatrixHypergraph, motifs::Vector{Vector{Tuple{Int32,Int32,Int32}}}, m::Int8; d=0.5, α=0.85, ϵ=1.0e-6)
    G = dyadic_projection(M)
    W = motif_cooccurence(M,motifs,m)
    #println(W)
    return motif_pagerank(Matrix(sparse(G)),Matrix(W); d=d, α=α, ϵ=ϵ)
end

"""
`n_argmax`
==========

Returns the indices of the top `n` maximal elements in descending order of array value.
"""
function n_argmax(A::Vector{T}, n::Int64) where T <: Real
    S = copy(A)
    inds = zeros(n)

    for i = 1:n
        max_ind = argmax(S)
        S[max_ind] = 0
        inds[i] = max_ind
    end

    return inds
end


#enron = read("C:\\Users\\Elizabeth Turner\\Documents\\Josh\\School\\F1\\Network Science\\HO-HyperRank-matrix\\src\\new-enron.txt")
#motifs = all_motifs(enron)
congress = read_arb("C:\\Users\\Elizabeth Turner\\Documents\\Josh\\School\\F1\\Network Science\\HO-HyperRank-matrix\\congress-bills")
motifs = all_motifs(congress)

#println(motif_cooccurence(enron,motifs,Int8(2)))
println(n_argmax(hyper_rank(congress, motifs, Int8(2); d=0.0),10))
#println(n_argmax(pagerank(dyadic_projection(enron),0.85),10))
#println(argmax(pagerank(dyadic_projection(enron),0.85)))
#Profile.init(delay=0.5)
#Juno.@profiler all_motifs(enron)
