include("Hypergraph.jl")
println(dyadic_projection(Hypergraph([[1,2,3], [2,3], [3,5,6], [4]], 7, 4))).weights) # Dyadic projection working!
