include("Hypergraph.jl")
using SparseArrays

"""
`all_motifs`
============

Classifies all triplets of edges in a given hypergraph into h-motifs.

Arguments
---------
    - `H::Hypergraph`: The hypergraph to analyze

Returns
-------
A dictionary that relates Boolean 7-lists to triplets of motifs. Each 7-list
represents an h-motif, and each associated triplet is an instance of that h-motif.
"""
function all_motifs(H::Hypergraph)
    motif_inst = Dict{Vector{Bool},Vector{Tuple{Int64,Int64,Int64}}}()

    for p1 in [[0,0,0],[0,0,1],[0,1,1],[1,1,1]] # Initialize all possible motif keys
        for p2 in [[0,0,0],[0,0,1],[0,1,1],[1,1,1]]
            for f in [[0],[1]]
                motif_inst[[p1;p2;f]] = []
            end
        end
    end

    for i = 1:H.m-2
        for j = i+1:H.m-1
            for k = j+1:H.m
                #=println((H.edges[i],H.edges[j],H.edges[k]))
                println(motif_vec(H.edges[i],H.edges[j],H.edges[k]))
                println()=#
                if (!isempty(intersect(H.edges[i], H.edges[j])) && !isempty(intersect(H.edges[j], H.edges[k]))) || # Triplet is connected
                   (!isempty(intersect(H.edges[i], H.edges[k])) && !isempty(intersect(H.edges[j], H.edges[k]))) ||
                   (!isempty(intersect(H.edges[i], H.edges[k])) && !isempty(intersect(H.edges[i], H.edges[j])))
                   push!(motif_inst[motif_vec(H.edges[i],H.edges[j],H.edges[k])], (i,j,k))
               end
            end
        end
    end

    return motif_inst
end

motif_vec(ei::Vector{Int64},ej::Vector{Int64},ek::Vector{Int64}) = # Turns a triplet of edges into the correct h-motif vector
    [sort([!isempty(setdiff(setdiff(ei,ej),ek)), !isempty(setdiff(setdiff(ej,ek),ei)), !isempty(setdiff(setdiff(ek,ei),ej))]);
     sort([!isempty(setdiff(intersect(ei,ej),ek)), !isempty(setdiff(intersect(ej,ek),ei)), !isempty(setdiff(intersect(ek,ei),ej))]);
     !isempty(intersect(ei,intersect(ej,ek)))]

function id_to_motif(id::Int8) # Given a h-motif id, returns the corresponding h-motif vector TODO

end

function motif_cooccurence(H::Hypergraph, m::Vector{Bool})
    W = spzeros(H.n, H.n)
    motifs = all_motifs(H)

    for t in motifs[m]
        for u in t[1]
            for v in t[2]
                for w in t[3]
                    W[u,v] += 1
                    W[v,u] += 1
                    W[v,w] += 1
                    W[w,v] += 1
                    W[u,w] += 1
                    W[w,u] += 1
                end
            end
        end
    end

    return W
end
