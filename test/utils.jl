import LazyDiffusions: invzero

@testset "invzero" begin
  n = 20
  v = sprand(n, 0.3)
  v = abs.(v)
  vt = deepcopy(v)
  for i = 1 : n
    if vt[i] > 0.0
      vt[i] = 1.0/vt[i] 
    end
  end

  iv = invzero(v) # test sparse call
  @test iv == vt

  v = Vector(v)
  vt = Vector(vt)
  iv = invzero(v) # test dense call
  @test iv == vt
end
