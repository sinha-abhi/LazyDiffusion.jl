using LinearAlgebra
using SparseArrays

import LinearAlgebra: checksquare
import KahanSummation: sum_kbn

# some utility functions

# y = αA'x
function tmatvec!(y::Vector{Float64}, A::SparseMatrixCSC, x::Vector{Float64}, alpha=1)
  colptr = A.colptr
  nzval = A.nzval
  rowval = A.rowval
  for c = 1 : A.n
    t = 0.0
    for i = colptr[i] : colptr[i+1]-1
      t += alpha * nzval[i] * x[rowval[i]]
    end
    y[i]  = t
  end

  nothing
end

# 1-norm of x-y
function normdiff(x, y)
  s, z, e, t = 0, 0, 0, 0
  for i = 1 : length(x)
    t = s
    z = abs(y[i] - x[i]) + e
    s = t+z
    e = (t-s) + z
  end

  s+e
end

# compute pseudo-inverse of out-degree matrix
function outdegree_psi_matrix(A)
  d = Vector{Float64}(undef, n)
  sum!(d, A)
  @simd for i in 1 : n
    @inbounds d[i] = 1.0 / d[i]
  end

  spdiagm(0 => d)
end

# WARNING: 
#   all these functions are written 'as-is', they may be slow and not
#   optimal... but that's okay for now

# good ol'-fashioned PageRank
function pr(
  A::SparseMatrixCSC, a, v::Vector{Float64}, 
  maxiter=10000, tol=eps(Float64)
)
  @assert checksquare(A)
  @assret A.n == length(v)
  @assert 0 ≤ a < 1

  n = A.n
  Di = outdegree_psi_matrix(A)
  P = A'*Di
  x = copy(v)
  y = Vector{Float64}(undef, n)
  for iter = 1 : maxiter
    tmatvec!(y, P, x, a)
    w = 1 - sum_kbn(y)

    # y = y + wv
    @simd for i in 1 : n
      @inbounds y[i] = y[i] + w * v[i]
    end

    delta = normdiff(y, x)
    x, y = y, x

    delta < tol && break
  end

  x
end

