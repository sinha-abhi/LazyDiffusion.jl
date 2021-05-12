# some utility functions

"""
Compute the 1-norm of the difference of two vectors with compensated summation.

  |y - x|_1
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
# TODO: split outdegree_pinv into two functions for: spare and dense
function outdegree_pinv(A::Union{SparseMatrixCSC{T}, Array{T}}) where T
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
function normalized_v(v::Vector{T}) where T
  _v = deepcopy(v)
  vsum = 1.0/KS.sum_kbn(_v)
  @inbounds for i in eachindex(_v)
    _v[i] *= vsum
  end

  _v
end

function normalized_v(v::SparseVector{T}) where T
  _v = deepcopy(v)
  nzv = nonzeros(_v)
  vsum = 1.0/KS.sum_kbn(nzv)
  @inbounds for i in eachindex(nzv)
    nzv[i] *= vsum 
  end

  _v
end

