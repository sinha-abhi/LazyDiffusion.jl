# some utility functions

"""
Compute the 1-norm of the difference of two vectors with compensated summation.
Equivalent to norm(x-y, 1), but more accurate.
"""
function normdiff(x::AbstractVector{T}, y::AbstractVector{T}) where T
  s, z, e, t = zero(T), zero(T), zero(T), zero(T)
  for i = 1 : length(x)
    t = s
    z = abs(y[i] - x[i]) + e
    s = t+z
    e = (t-s) + z
  end

  s+e
end

"""
Compute the pseudo-inverse (from GvL) of the out-degree matrix of the graph 
defined by A.
"""
function outdegree_pinv(A::MatrixUnion{T}) where T
  d = Vector{T}(undef, min(size(A)...))
  sum!(d, A)
  @simd for i = 1 : length(d)
    @inbounds if d[i] > zero(T)
      d[i] = one(T) / d[i] 
    end
  end

  return if A isa SparseMatrixCSC
    spdiagm(0 => d)
  else
    Diagonal(d)
  end
end

"""
Normalize the sum of v to 1.
"""
function normalize_pref_vec(v::Vector{T}) where T
  _v = deepcopy(v)
  vsum = 1.0/KS.sum_kbn(_v)
  @inbounds for i in eachindex(_v)
    _v[i] *= vsum
  end

  _v
end

function normalize_pref_vec(v::SparseVector{T}) where T
  _v = deepcopy(v)
  nzv = nonzeros(_v)
  vsum = 1.0/KS.sum_kbn(nzv)
  @inbounds for i in eachindex(nzv)
    nzv[i] *= vsum 
  end

  _v
end

"""
Compute the inverse of all nonzero elements in v.
"""
function invzero(v::Vector{T}) where T
  n = length(v)
  invv = Vector{T}(undef, n)
  @simd for i = 1 : n
    if v[i] > zero(T)
      @inbounds invv[i] = one(T) / v[i]
    else
      invv[i] = zero(T)
    end
  end

  invv
end

function invzero(v::SparseVector{T}) where T
  nz_ind = findall(!iszero, v)
  nnz = length(nz_ind)
  invv = Vector{T}(undef, nnz)

  for (i, nzind) in enumerate(nz_ind)
    @inbounds invv[i] = one(T) / v[nzind]
  end

  sparsevec(nz_ind, invv, length(v))
end

