using Test

using LazyDiffusions
using LazyDiffusions.PRAlgs

@testset verbose=true "LazyDiffusions" begin
  include("pagerank.jl")
  include("utils.jl")
end
