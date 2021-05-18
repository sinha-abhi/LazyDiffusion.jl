# Implementations of various PageRank variations
module PageRank

using LinearAlgebra
using LinearAlgebra.BLAS
using SparseArrays

import KahanSummation as KS
import ..LazyDiffusions: 
  MatrixUnion, VectorUnion,
  checksquare, invzero, normalize_pref_vec, normdiff, outdegree_pinv

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
Base.@kwdef mutable struct PROptions{T <: Real}
  n::Int
  α::Float64                           = 0.85
  tol::Float64                         = 1e-7
  v::VectorUnion{T}                    = ones(T, n)/n
  maxiter::Int                         = 500
  x0::VectorUnion{T}                   = deepcopy(v)
  alg::PRAlgs.PRAlg                    = PRAlgs.POWER
  approx_bp::Float64                   = 1e-3
  approx_boundary::Union{Int, Float64} = Inf
  approx_subiter::Int                  = 5
end

"""
Compute the personalized PageRank vector for a directed graph A.

The out-bound edges should be represented in the rows of A.
"""
function pagerank(A::MatrixUnion{T}, opts::PROptions{T}) where T
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

# power method for personalized PageRank 
function power_pagerank(P::MatrixUnion{T}, opts::PROptions{T}) where T
  x = deepcopy(opts.x0)
  α = opts.α
  v = normalize_pref_vec(opts.v) # get normalized v
  tol = opts.tol
  maxiter = opts.maxiter

  y = Vector{T}(undef, length(x))
  for _ = 1 : maxiter
    y = α*(x'*P)'
    ω = 1 - KS.sum_kbn(y)
    axpy!(ω, v, y)
    δ = normdiff(x, y)
    x = y
    δ < tol && break
  end

  x
end

# approximate personalized PageRank (from Gleich and Polito)
#   restricted personalized PageRank,
#   boundary restricted personalized PageRank
function approx_pagerank(P::MatrixUnion{T}, opts::PROptions{T}) where T
  α = opts.α
  v = normalize_pref_vec(opts.v)
  n = opts.n
  tol = opts.tol
  maxiter = opts.maxiter
  subiter = opts.approx_subiter
  bp = opts.approx_bp
  boundary = opts.approx_boundary

  # find the seed pages
  p = findall(!iszero, v)
  nnz = length(p)
  x = ones(T, nnz) / nnz

  loc = Int[]
  active = p
  frontier = p
  for _ = 1 : maxiter
    local y
    if boundary == 1
      sp = sortperm(-x)
      cs = cumsum(x[sp])
      spactive = active[sp]
      allexpand_ind = cs .< (1-bp)
      # include the first 0 since we want cs > 1-bp
      allexpand_ind[findfirst(iszero, allexpand_ind)] = 1
      allexpand = spactive[allexpand_ind]
      toexpand = setdiff(allexpand, loc)
    else
      # expand all pages with sufficient tolerance
      allexpand = active[x .> bp]
      toexpand = setdiff(allexpand, loc)
    end

    xp = zeros(n)
    xp[[loc; frontier]] = x
    if length(toexpand) > 0
      loc = [loc; toexpand]
      nzls = LinearIndices(findall(!iszero, sum(P[loc,:], dims=1)))
      frontier = setdiff(nzls, loc)
      active = [loc; frontier]
    else
      x = xp[loc]
    end

    Lp = P[loc, active]
    outdegree = sum(Lp, dims=2)
    outdegree = vec([outdegree; zeros(T, length(frontier))])

    L = [Lp; spzeros(length(frontier), length(active))]
    x2 = [x; xp[frontier]]
    for _ = 1 : subiter
      y = α*L'*(invzero(outdegree) .* x2)
      ω = 1 - KS.sum_kbn(y)
      y[1:length(p)] = y[1:length(p)] + ω*v
      x2 = y
    end

    x2 = [x; xp[frontier]]
    δ = normdiff(y, x2)
    x = y
    δ < tol && break
  end

  x
end

end # module