include("../src/Dct2.jl")
include("../src/Utils.jl")

using LinearAlgebra
using .Dct2
using .Utils
using Plots
using FFTW


function TestTimeDct()
    dims = 4:1000:4000
    time_dct = []
    time_fft = []
    for i in dims
        M = Utils.gen_random_matrix(i, i)
        push!(time_dct, @elapsed Dct2.DctII(M))
        push!(time_fft, @elapsed FFTW.dct(M))
        println(time_dct)
        println(time_fft)
    end
    plot(dims, [time_dct, time_fft])
end


input = Float64[231, 32, 233, 161, 24, 71, 140, 245]
input2 = Float64[
    231 32 233 161 24 71 140 245;
    247 40 248 245 124 204 36 107;
    234 202 245 167 9 217 239 173;
    193 190 100 167 43 180 8 70;
    11 24 210 177 81 243 8 112;
    97 195 203 47 125 114 165 181;
    193 70 174 167 41 30 127 245;
    87 149 57 192 65 129 178 228
]
println(typeof(input2))
println(sum(FFTW.dct(input) - Dct2.Dct(input)))
println(sum(FFTW.dct(input2) - Dct2.DctII(input2)))

#TestTimeDct()