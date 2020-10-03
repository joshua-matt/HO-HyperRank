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
    motif_inst = Dict{Int8,Vector{Tuple{Int64,Int64,Int64}}}(i => [] for i = 0:26)

    for i = 1:H.m-2
        for j = i+1:H.m-1
            for k = j+1:H.m
                #=println((H.edges[i],H.edges[j],H.edges[k]))
                println(motif_vec(H.edges[i],H.edges[j],H.edges[k]))
                println()=#
                if (!isempty(intersect(H.edges[i], H.edges[j])) && !isempty(intersect(H.edges[j], H.edges[k]))) || # Triplet is connected
                   (!isempty(intersect(H.edges[i], H.edges[k])) && !isempty(intersect(H.edges[j], H.edges[k]))) ||
                   (!isempty(intersect(H.edges[i], H.edges[k])) && !isempty(intersect(H.edges[i], H.edges[j])))
                   push!(motif_inst[get_id(H.edges[i],H.edges[j],H.edges[k])], (i,j,k))
               end
            end
        end
    end

    return motif_inst
end

function get_id(ei::Vector{Int64}, ej::Vector{Int64}, ek::Vector{Int64})
	motif_id::Vector{Int8} = [ # Used for converting from binary vector to motif id
		0, 0, 0, 0, 0, 0, 17, 7,
		0, 0, 17, 7, 17, 7, 23, 13,
		0, 0, 0, 3, 0, 3, 18, 8,
		0, 0, 19, 9, 19, 9, 24, 14,
		0, 0, 0, 0, 0, 3, 19, 9,
		0, 3, 19, 9, 18, 8, 24, 14,
		0, 1, 0, 5, 0, 4, 20, 10,
		0, 5, 21, 11, 20, 10, 25, 15,
		0, 0, 0, 3, 0, 0, 19, 9,
		0, 3, 18, 8, 19, 9, 24, 14,
		0, 1, 0, 4, 0, 5, 20, 10,
		0, 5, 20, 10, 21, 11, 25, 15,
		0, 1, 0, 5, 0, 5, 21, 11,
		0, 4, 20, 10, 20, 10, 25, 15,
		0, 2, 0, 6, 0, 6, 22, 12,
		0, 6, 22, 12, 22, 12, 26, 16
	]

	vect = (!isempty(setdiff(setdiff(ei,ej))) << 6) +
		   (!isempty(setdiff(setdiff(ej,ek),ei)) << 5) +
		   (!isempty(setdiff(setdiff(ek,ei),ej)) << 4) +
		   (!isempty(setdiff(intersect(ei,ej),ek)) << 3) +
		   (!isempty(setdiff(intersect(ej,ek),ei)) << 2) +
		   (!isempty(setdiff(intersect(ek,ei),ej)) << 1) +
		   (!isempty(intersect(ei,intersect(ej,ek))))
	println(motif_id[vect+1])
	return motif_id[vect+1]
end

function motif_cooccurence(H::Hypergraph, m::Int8)
    W = spzeros(H.n, H.n)
    motifs = all_motifs(H)

    for tr in motifs[m] # TODO: Only connect adjacent nodes? So if we have an open motif, we don't connect nodes at opposite ends
		nodes = Set(union(H.edges[tr[1]], H.edges[tr[2]], H.edges[tr[3]]))
		len = length(nodes)
        for u = 1:len-1
			for v = u+1:len
				W[u,v] += 1
				W[v,u] += 1
			end
		end
    end

    return W
end
