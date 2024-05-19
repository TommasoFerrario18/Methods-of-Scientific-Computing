include("../src/Dct2.jl")
include("../src/Utils.jl")

using LinearAlgebra
using .Dct2
using .Utils
using Plots
using FFTW



function TestDct()

    par(x) = -x^2 + 4x
    
    input = Float64[]
    for i in 0:0.1:4
        push!(input, par(i))
    end


    for i in 0:0.1:4
        push!(input, par(4-i))
    end
    
    binput = bar(input, title="Target Function")
    savefig(binput,"target.png")
    
    
    base = Dct2.Gen_ortogonal_cos_base(length(input))
    coef = Dct2.Get_coefficients(input, base)
    
    
    bcoef = bar(coef, title="Coefficients computed in a custom way")
    savefig(bcoef,"dct_coeff_custom.png")
    
    lcoef = FFTW.dct(input)
    blcoef = bar(lcoef, title="Coefficients computed using std library")
    savefig(blcoef,"dct_coeff_library.png")
    
    btargetlibrary = bar(transpose(base) * lcoef, title="Approx Target Function using custom coefficients")
    savefig(btargetlibrary,"target_library.png")
    
    btargetcustom = bar(transpose(base) * coef, title="Approx Target Function using std library coefficients")
    savefig(btargetcustom,"target_custom.png")

    println(norm(coef - lcoef))
end

TestDct()

input = Float64[231, 32, 233, 161, 24, 71, 140, 245]

println(FFTW.dct(input))
println(Dct2.Dct(input))
