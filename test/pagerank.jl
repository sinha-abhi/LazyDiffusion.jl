using LinearAlgebra
using SparseArrays

@testset "pagerank" begin
  n = 10
  A = sparse(1.0I, n, n)
  v = rand(n)
  vn = v / sum(v)
  opts = PROptions{eltype(A)}(n=n, v=v)
  x = pagerank(A, opts)
  @test norm(x-vn, 1) <= n*eps(Float64)

  opts = PROptions{eltype(A)}(n=n, v=v, x0=v)
  x = pagerank(A, opts)
  @test norm(x-vn, 1) <= n*eps(Float64)
end
