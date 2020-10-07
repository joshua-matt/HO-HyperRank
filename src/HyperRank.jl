include("Hypergraph.jl")
#include("weighted_pagerank.jl")
include("h_motifs.jl")
using SparseArrays, LinearAlgebra
using MatrixNetworks
using BenchmarkTools

function motif_pagerank(G::Matrix, M::Matrix; d=0.5, α=0.85, n::Integer=100, ϵ=1.0e-6) # TODO: Find way to add sparse matrices
    return pagerank(MatrixNetwork(sparse((d*G + (1-d)*M))), α, n, ϵ)
end

"""
`hyper_rank`
============


"""
function hyper_rank(M::MatrixHypergraph, m::Int8; d=0.5, α=0.85, n::Integer=100, ϵ=1.0e-6)
    G = dyadic_projection(M)
    W = Matrix(motif_cooccurence(M,m))
    return motif_pagerank(Matrix(G.weights),W; d=d, α=α, n=n, ϵ=ϵ)
end

enron = MatrixHypergraph("C:\\Users\\divin\\OneDrive\\Documents\\School\\F1\\Network Analysis\\Project\\data\\new-enron.txt")
hyper_rank(enron, Int8(13))
#using Profile
#Profile.init(delay=0.5)
#Juno.@profiler get_hyperwedges(enron)
#@time get_hyperwedges(enron)
#println(size(enron.incidence))
#@benchmark println(hyper_rank(enron, Int8(6)))

#for i = 1:26
#    println(i,hyper_rank(G, Int8(i); d=0., α=1.))
#end

#=G = dyadic_projection(Hypergraph([[1,2,3], [2,3], [3,5,6], [4]], 7, 4))
M = dyadic_projection(Hypergraph([[1,3,4], [2,3], [3,5,6], [4]], 7, 4))
for k = 0:0.1:1
    println(motif_pagerank(Matrix(G.weights), Matrix(G.weights); d=k, α=1.))
end
#println(weight(G,3))=#

# To quickly find wedges, we can use a modified Chiba-Nishizeki, where we check highest degree first,
# We could just not allow open motifs... In general, it looks like open motifs are not significant
