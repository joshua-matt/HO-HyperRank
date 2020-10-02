include("Hypergraph.jl")
include("weighted_pagerank.jl")
include("h_motifs.jl")
using SparseArrays

function motif_pagerank(G::Matrix, M::Matrix; d=0.5, α=0.85, n::Integer=100, ϵ=1.0e-6) # TODO: Find way to add sparse matrices
    return pagerank(SimpleWeightedGraph(sparse((d*G + (1-d)*M))), α, n, ϵ)
end

"""
`hyper_rank`
============


"""
function hyper_rank(H::Hypergraph, m::Int8; d=0.5, α=0.85, n::Integer=100, ϵ=1.0e-6)
    G = dyadic_projection(H)
    W = Matrix(motif_cooccurence(H,m))
    return motif_pagerank(Matrix(G.weights),W; d=d, α=α, n=n, ϵ=ϵ)
end

G = Hypergraph([[1,2,3],[2,3,4],[4,5,6]]) # TODO: Error with excluding nodes from motif matrix

for i = 1:26
    println(i,hyper_rank(G, Int8(i); d=0., α=1.))
end

#=G = dyadic_projection(Hypergraph([[1,2,3], [2,3], [3,5,6], [4]], 7, 4))
M = dyadic_projection(Hypergraph([[1,3,4], [2,3], [3,5,6], [4]], 7, 4))
for k = 0:0.1:1
    println(motif_pagerank(Matrix(G.weights), Matrix(G.weights); d=k, α=1.))
end
#println(weight(G,3))=#
