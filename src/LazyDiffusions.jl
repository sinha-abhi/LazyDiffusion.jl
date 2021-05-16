module LazyDiffusions

using LinearAlgebra
using LinearAlgebra.BLAS
using SparseArrays

import LinearAlgebra: checksquare
import KahanSummation as KS

MatrixUnion{T} = Union{SparseMatrixCSC{T}, Array{T}} where T
VectorUnion{T} = Union{SparseVector{T}, Vector{T}} where T


include("pagerank.jl")
include("utils.jl")

export
  PROptions,
  pagerank

end # module
