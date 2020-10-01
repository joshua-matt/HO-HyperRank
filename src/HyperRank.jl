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
function hyper_rank(H::Hypergraph, m::Vector{Bool}; d=0.5, α=0.85, n::Integer=100, ϵ=1.0e-6)
    G = dyadic_projection(H)
    W = Matrix(motif_cooccurence(H,m))
    return motif_pagerank(Matrix(G.weights),W; d=d, α=α, n=n, ϵ=ϵ)
end

println(hyper_rank(Hypergraph([[1,2,3], [2,3], [3,5,6], [4]], 7, 4), [false,true,true,false,false,true,true]; d=0.5))

#=G = dyadic_projection(Hypergraph([[1,2,3], [2,3], [3,5,6], [4]], 7, 4))
M = dyadic_projection(Hypergraph([[1,3,4], [2,3], [3,5,6], [4]], 7, 4))
for k = 0:0.1:1
    println(motif_pagerank(Matrix(G.weights), Matrix(G.weights); d=k, α=1.))
end
#println(weight(G,3))=#
