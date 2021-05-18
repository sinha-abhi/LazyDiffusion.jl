module LazyDiffusions

using LinearAlgebra
using LinearAlgebra.BLAS
using SparseArrays

import LinearAlgebra: checksquare
import KahanSummation as KS

MatrixUnion{T <: Real} = Union{SparseMatrixCSC{T}, Array{T}} where T <: Real
VectorUnion{T <: Real} = Union{SparseVector{T}, Vector{T}} where T <: Real

include("utils.jl")

# submodules
include("diffusions.jl")
include("pagerank.jl")

using .PageRank.PRAlgs: PRAlg
using .PageRank: PROptions, pagerank
export
  PRAlg,
  PROptions,
  pagerank

end # module
