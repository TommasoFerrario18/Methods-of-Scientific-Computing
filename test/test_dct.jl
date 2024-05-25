include("../src/Dct2.jl")
include("../src/Utils.jl")

using LinearAlgebra
using .Dct2
using .Utils
using Plots
using FFTW


function TestTimeDct()
    start_dim = 40
    max_iter = 5
    list_dims = []
    time_dct = []
    time_fft = []
    for i in 1:max_iter
        M = Utils.GenRandomMatrix(start_dim, start_dim)
        push!(time_dct, @elapsed Dct2.DctII(M))
        push!(time_fft, @elapsed FFTW.dct(M))
        push!(list_dims, start_dim)
        start_dim *= 2
    end
    println(time_dct)
    println(time_fft)
    println(list_dims)
    x = range(list_dims[1], list_dims[end], length=1000)
    #x = range(0.00001, 0.1, length=1000)
    time_theoretical_custom = []
    time_theoretical_library = []
    for i in collect(x)
        push!(time_theoretical_custom, i^3)
        push!(time_theoretical_library, i^2 * log10(i))
    end
    p = plot(list_dims, time_dct, label="dct2 custom", lc=:blue)
    plot!(p, x, time_theoretical_custom, label="dct2 custom theoretical time ", lc=:blue, linestyle=:dash)
    plot!(p, list_dims, time_fft, label="dct2 library", lc=:orange)
    plot!(p, x, time_theoretical_library, label="dct2 library theoretical time ", lc=:orange, linestyle=:dash)
    plot!(p, xscale=:log10, yscale=:log10, minorgrid=true)
    title!(p, "Time of dct2 execution")
    xlabel!(p, "matrix dimension")
    ylabel!(p, "time (s)")
    savefig(p, "times_plot.png")
    println(time_theoretical_custom)
    println(time_theoretical_library)
end


# input = Float64[231, 32, 233, 161, 24, 71, 140, 245]
# input2 = Float64[
#     231 32 233 161 24 71 140 245;
#     247 40 248 245 124 204 36 107;
#     234 202 245 167 9 217 239 173;
#     193 190 100 167 43 180 8 70;
#     11 24 210 177 81 243 8 112;
#     97 195 203 47 125 114 165 181;
#     193 70 174 167 41 30 127 245;
#     87 149 57 192 65 129 178 228
# ]
# println(typeof(input2))
# println(sum(FFTW.dct(input) - Dct2.Dct(input)))
# println(sum(FFTW.dct(input2) - Dct2.DctII(input2)))

TestTimeDct()