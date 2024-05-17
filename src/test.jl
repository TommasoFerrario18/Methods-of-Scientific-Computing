include("Dct2.jl")
include("Utils.jl")

using LinearAlgebra
using .Dct2
using .Utils
using Plots
using FFTW

dim = 3

m = Utils.gen_random_matrix(dim, dim)

c1 = Dct2.Dct(m[1,:])
"""
c2 = Dct2.Dct(m[2,:])
c3 = Dct2.Dct(m[3,:])
"""
println("ris")
println(m[1,:])
println(rfft(m[1,:]))
println(c1)