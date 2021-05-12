# Implementations of various PageRank variations

# supported PageRank variations
baremodule PRAlgs
using Base: @enum

@enum PRAlg begin
  STANDARD
end

end

struct PRProblem{T}
  A::Union{SparseMatrixCSC{T}, Array{T}}
  α::T
  v::Union{SparseVector{T}, Vector{T}}
  n::Int
end

function PRProblem{T}(
  A::Union{SparseMatrixCSC{T}, Array{T}}, 
  α::T, 
  v::Union{SparseVector{T}, Vector{T}}
) where T
  checksquare(A)
  A.n == length(v) || throw(DimensionMismatch(
    "expected v of length $(A.n), but got $(length(v))"
  ))
  0 <= α < 1 || throw(DimensionMismatch("expected α ∈ [0,1), but got α = $α"))

  PRProblem{T}(A, α, v, A.m)
end

function pagerank(
  p::PRProblem{T}; 
  alg::PRAlgs.PRAlg, maxiter=10000, tol=1e-7
) where T
  alg == PRAlgs.STANDARD && standard_pagerank(p, maxiter, tol)
end

## pagerank variations

function standard_pagerank(p::PRProblem{T}, maxiter, tol) where T
  P = outdegree_pinv(p.A) * p.A # normalize A
  v = normalized_v(p.v) # get normalized v
  x = deepcopy(v)
  y = Vector{T}(undef, p.n)
  for iter = 1 : maxiter
    y = p.α*(x'*P)'
    ω = 1 - KS.sum_kbn(y)
    y = ω*v + y
    δ = normdiff(x, y)
    x = y
    δ < tol && break
  end

  x
end
