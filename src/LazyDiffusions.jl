module LazyDiffusions

using LinearAlgebra
using LinearAlgebra.BLAS
using SparseArrays

import LinearAlgebra: checksquare
import KahanSummation as KS

MatrixUnion{T <: Real} = Union{SparseMatrixCSC{T}, Array{T}} where T <: Real
VectorUnion{T <: Real} = Union{SparseVector{T}, Vector{T}} where T <: Real

include("pagerank.jl")
include("utils.jl")

export
  PROptions,
  pagerank

end # module
