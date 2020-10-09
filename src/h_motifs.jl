include("Hypergraph.jl")
using SparseArrays

const motif_id = [ # Used for converting from binary vector to motif id
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

"""
`all_motifs`
============

Classifies all triplets of edges in a given hypergraph into h-motifs.

Arguments
---------
    - `H::Hypergraph`: The hypergraph to analyze

Returns
-------
A 2D-array where A[i] is the list of triplets that are instances of h-motif i.
"""

function all_motifs(M::MatrixHypergraph)
	motif_inst::Vector{Vector{Tuple{Int32,Int32,Int32}}} = [[] for i = 0:26]
	adj = dyadic_projection(dual(M))
	inc = M.incidence'
	m = size(inc,1)
	edges = [inc[i,:].nzind for i = 1:m]
	sizes = length.(edges)
	inters::Dict{Tuple{Int32,Int32},Vector{Int32}} = Dict((i,j) => intersect(edges[i],edges[j]) for i = 1:m for j in adj.ci[adj.rp[i]:adj.rp[i+1]-1] if i < j)

	for a = 1:m
		size_a = sizes[a]
		deg_a = adj.rp[a+1]-adj.rp[a]
		neigh = adj.ci[adj.rp[a]:adj.rp[a+1]-1]
		for hb = 1:deg_a-1
			b = neigh[hb]
			if a > b break end
			size_b = sizes[b]
			deg_b = adj.rp[b+1]-adj.rp[b]

			for hc = hb+1:deg_a
				c = neigh[hc]
				size_c = sizes[c]
				deg_c = adj.rp[c+1]-adj.rp[c]

				if length(get(inters,(b,c),Int32[])) > 0
					push!(motif_inst[
								get_id(size_a,size_b,size_c,length(inters[(a,b)]),length(inters[(b,c)]),length(inters[(a,c)]),length(intersect(inters[(a,b)],inc[c,:].nzind)))+1
									], (a,b,c))
				else
					push!(motif_inst[
								get_id(size_a,size_b,size_c,length(inters[(a,b)]),0,length(inters[(a,c)]),0)+1
									], (a,b,c))
				end
			end
		end
	end

	return motif_inst
end

function get_id(d_a::Int64,d_b::Int64,d_c::Int64,
				i_ab::Int64,i_bc::Int64,i_ca::Int64,
				g_abc::Int64)
	a = d_a - (i_ab+i_ca) + g_abc
	b = d_b - (i_bc+i_ab) + g_abc
	c = d_c - (i_ca+i_bc) + g_abc
	d = i_ab - g_abc
	e = i_bc - g_abc
	f = i_ca - g_abc
	g = g_abc

	vect = ((a > 0) << 6) + ((b > 0) << 5) + ((c > 0) << 4) + ((d > 0) << 3) +
		   ((e > 0) << 2) + ((f > 1) << 1) + ((g > 0) << 0)
	return motif_id[vect+1]
end

function motif_cooccurence(M::MatrixHypergraph, m::Int8)
    W = spzeros(Int32, M.n, M.n)
    motifs = all_motifs(M)
	inc = M.incidence

    for t in motifs[m] # TODO: Only connect adjacent nodes? So if we have an open motif, we don't connect nodes at opposite ends
		i,j,k = t
		nodes = Set(union(inc[:,i], inc[:,j], inc[:,k]))
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
