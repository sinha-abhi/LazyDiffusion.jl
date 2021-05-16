# Implementations of various PageRank variations

# supported PageRank variations
baremodule PRAlgs
using Base: @enum

@enum PRAlg begin
  POWER 
  APPROX 
end
end # baremodule

"""
PROptions
  n: PageRank problem size
  α: teleporation parameter
  tol: stopping tolerance
  v: preference (or personalization vector)
  maxiter: maximum number of iterations
  x0: initial vector
  alg: algorithm type
  approx_bp: boundary probability to expand
  approx_subiter: number of subiterations of power iterations
"""
Base.@kwdef mutable struct PROptions{T}
  n::Int
  α::Float64                           = 0.85
  tol::Float64                         = 1e-7
  v::VectorUnion{T}                    = ones(n)/n
  maxiter::Int                         = 500
  x0::VectorUnion{T}                   = deepcopy(v)
  alg::PRAlgs.PRAlg                    = PRAlgs.POWER
  approx_bp::Float64                   = 1e-3
  approx_boundary::Union{Int, Float64} = Inf
  approx_subiter::Int                  = 5
end

function pagerank(A::MatrixUnion{T}, opts::PROptions) where T
  checksquare(A)
  A.n == length(opts.v) || throw(DimensionMismatch(
    "expected v of length $(A.n), but got $(length(opts.v))"
  ))
  0 <= opts.α < 1 || throw(DimensionMismatch("expected α ∈ [0,1), but got α = $(opts.α)"))

  P = outdegree_pinv(A) * A # normalize A

  opts.alg == PRAlgs.POWER && return power_pagerank(P, opts)
  opts.alg == PRAlgs.APPROX && return approx_pagerank(P, opts)
end

## pagerank variations

# power method for pagerank
function power_pagerank(P::MatrixUnion{T}, opts::PROptions{T}) where T
  x = deepcopy(opts.x0)
  α = opts.α
  v = normalize_pref_vec(opts.v) # get normalized v
  tol = opts.tol
  maxiter = opts.maxiter

  y = Vector{T}(undef, length(x))
  for iter = 1 : maxiter
    y = α*(x'*P)'
    ω = 1 - KS.sum_kbn(y)
    axpy!(ω, v, y)
    δ = normdiff(x, y)
    x = y
    δ < tol && break
  end

  x
end

# approximate pagerank
function approx_pagerank(P::MatrixUnion{T}, opts::PROptions{T}) where T
  # find the seed pages
  p = findall(!iszero, v)
  nnz = length(p)
  x = ones(nnz) / nnz

  loc = Int[]
  active = p
  frontier = p
  for iter = 1 : maxiter
    # TODO: finish implementing
    # sp = sortperm(-x)

    δ < tol && break
  end

  nothing
end
