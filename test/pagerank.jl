using LinearAlgebra
using SparseArrays

@testset "power_pr" begin
  n = 10
  A = sparse(1.0I, n, n)
  v = rand(n)
  vn = v / sum(v)
  opts = PROptions{eltype(A)}(n=n, v=v)
  x = pagerank(A, opts)
  @test norm(x-vn, 1) <= n*eps(Float64)

  opts.x0 = v
  x = pagerank(A, opts)
  @test norm(x-vn, 1) <= n*eps(Float64)
end

@testset "approx_pr" begin
  n = 10
  A = sparse(1.0I, n, n)
  v = rand(n)

  opts = PROptions{eltype(A)}(n=n, v=v, alg=PRAlgs.APPROX)
  x = pagerank(A, opts)

  opts = PROptions{eltype(A)}(n=n, v=v)
  x2 = pagerank(A, opts)
  @test isapprox(norm(x-x2, 1), 0.0, atol=1e-7)
end
