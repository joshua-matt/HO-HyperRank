"""
Finding and classifying h-motifs. Based on code from https://github.com/geonlee0325/MoCHy.
"""

include("Hypergraph.jl") # Used for working with hypergraphs
using SparseArrays # Used to construct the motif co-occurrence matrix

const motif_id = [ # Used for converting from three-way Venn to motif id
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
A 2D-array where `A[i]` is the list of triplets that are instances of h-motif `i`.
"""

function all_motifs(M::MatrixHypergraph)
	motif_inst::Vector{Vector{Tuple{Int64,Int64,Int64}}} = [[] for i = 0:26] # Initialize the motif list
	G = dyadic_projection(dual(M))
	adj = [G.ci[G.rp[i]:G.rp[i+1]-1] for i = 1:G.n] # Get which hyperedges are adjacent
	inc = M.incidence'
	m = size(inc,1)
	edges = [inc[i,:].nzind for i = 1:m] # Get the nodes involved in each hyperedge
	sizes = length.(edges)
	inters::Dict{Tuple{Int64,Int64},Vector{Int64}} = Dict((i,j) => intersect(edges[i],edges[j]) for i = 1:m for j in adj[i] if i < j) # Compute the intersection between hyperedges. This is the primary bottleneck

	for a = 1:m # Enumerate all hyperedges
		size_a = sizes[a]
		deg_a = G.rp[a+1]-G.rp[a]
		neigh = adj[a]
		for hb = 1:deg_a-1 # Only look at neighbors, so that we only consider connected triplets
			b = neigh[hb]
			if a > b break end # Only considers unique triplets
			size_b = sizes[b]
			deg_b = G.rp[b+1]-G.rp[b]

			for hc = hb+1:deg_a
				c = neigh[hc]
				size_c = sizes[c]
				deg_c = G.rp[c+1]-G.rp[c]

				if length(get(inters,(b,c),Int64[])) > 0 # Save time by not computing three-way intersection if it doesn't exist
					id = get_id(size_a,size_b,size_c,
								length(inters[(a,b)]),length(inters[(b,c)]),length(inters[(a,c)]),
								length(intersect(inters[(a,b)],edges[c]))) + 1
					push!(motif_inst[id], (a,b,c))
				else
					id = get_id(size_a,size_b,size_c,
								length(inters[(a,b)]),0,length(inters[(a,c)]),
								0)+1
					push!(motif_inst[id], (a,b,c))
				end
			end
		end
	end

	return motif_inst
end

"""
`get_id`
========

Classifies which h-motif a triplet of hyperedges is an instance of based on information
about their sizes and intersections.

Arguments
---------
	- `d_a::Int64`: The size of the first hyperedge
	- `d_b::Int64`: The size of the second hyperedge
	- `d_c::Int64`: The size of the third hyperedge
	- `i_ab::Int64`: The size of the intersection between the first two hyperedges
	- `i_bc::Int64`: The size of the intersection between the last two hyperedges
	- `i_ca::Int64`: The size of the intersection between the first and third hyperedges
	- `g_abc::Int64`: The size of the three-way intersection among all hyperedges

Returns
-------
The motif that the triplet is an instance of. Returns 0 if the motif is invalid.
The diagrams corresponding to the id's may be found on page 3 of https://arxiv.org/pdf/2003.01853.pdf.
"""
function get_id(d_a::Int64,d_b::Int64,d_c::Int64,
				i_ab::Int64,i_bc::Int64,i_ca::Int64,
				g_abc::Int64)
	a = d_a - (i_ab+i_ca) + g_abc # Number of nodes unique to first hyperedge
	b = d_b - (i_bc+i_ab) + g_abc
	c = d_c - (i_ca+i_bc) + g_abc
	d = i_ab - g_abc # Number of nodes shared only between first and second hyperedges
	e = i_bc - g_abc
	f = i_ca - g_abc
	g = g_abc # Number of nodes shared among all three

	vect = ((a > 0) << 6) + ((b > 0) << 5) + ((c > 0) << 4) + ((d > 0) << 3) +
		   ((e > 0) << 2) + ((f > 1) << 1) + ((g > 0) << 0) # Encoding the Venn diagram as a binary number
	return motif_id[vect+1] # Return the id located at that number (+1 because Julia is one-indexed)
end

"""
`motif_coocurrence`
===================

Constructs the h-motif co-occurrence matrix for a hypergraph, based on a particular
h-motif. The motif co-occurrence matrix records how many h-motif instances each
pair of nodes co-occurs in.

Arguments
---------
	- `M::MatrixHypergraph`: The hypergraph for which to create the matrix.
	- `motifs::Vector{Vector{Tuple{Int64,Int64,Int64}}}`: The classification of triplets of hyperedges
														  into h-motifs. A[i+1] is all instances of the ith motif,
														  with invalid instances occupying A[1].
	- `m::Int8`: The motif that the co-occurrence matrix is based on.

Returns
-------
A symmetric `SparseMatrix` where the ij entry is the number of instances of h-motif
`m` that nodes `i` and `j` co-occur in. If the two nodes do not share a hyperedge in
`M`, the entry will be 0.
The diagrams corresponding to the id's may be found on page 3 of https://arxiv.org/pdf/2003.01853.pdf.
"""
function motif_cooccurrence(M::MatrixHypergraph, motifs::Vector{Vector{Tuple{Int64,Int64,Int64}}}, m::Int8)
	inc = M.incidence'
	W = spzeros(Float64, size(inc,2), size(inc,2))

    for t in motifs[m+1]
		i,j,k = t
		co::Set{Set{Int64,Int64}} = Set() # Node pairs that co-occur together in the hyperedges of t
		# Only strengthen edges that already exist, and only strengthen once per motif (i.e. don't double-count edges on the overlap)
		for e in [inc[i,:].nzind, inc[j,:].nzind, inc[k,:].nzind]
			len = length(e)
	        for u = 1:len-1
				for v = u+1:len
					push!(co, (e[u],e[v]))
				end
			end
		end
		for p in co # Increase weight by one for each pair of hyperedge-sharing, motif co-occurring nodes
			W[p[1],p[2]] += 1
			W[p[2],p[1]] += 1
		end
    end
    return W
end
