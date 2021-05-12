module LazyDiffusions

using LinearAlgebra
using LinearAlgebra.BLAS
using SparseArrays

import LinearAlgebra: checksquare
import KahanSummation as KS

include("pagerank.jl")

include("utils.jl")

export
  PRProblem,
  pagerank

end # module
