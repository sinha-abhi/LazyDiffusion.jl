using LinearAlgebra
using SparseArrays

@testset "pagerank" begin
  n = 10
  A = sparse(1.0I, n, n)
  v = rand(n)
  vn = v / sum(v)
  p = PRProblem{Float64}(A, 0.85, v)
  x = pagerank(p, alg=PRAlgs.STANDARD)
  @test norm(x-vn, 1) <= n*eps(Float64)
end
