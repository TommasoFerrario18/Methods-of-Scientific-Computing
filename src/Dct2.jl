
module Dct2

using LinearAlgebra
using Plots
using FileIO
using Images



function Gen_ortogonal_cos_base(dim::Integer)::Matrix{Float64}
    base = zeros(dim,dim)
    

    for k in 1:dim 
        for i in 1:dim
            base[i, k] = cos((k-1) * pi * (2 * (i-1) + 1) / (2 * dim))
            #base[i, k] = cos((k-1) * pi * ((i-1) + 1/2)/ dim) libreria
        end
    end

    return base
end

function Test_ortogonal_cos_base(mat::Matrix{Float64})::Bool
    N = size(mat)[1]
    #sum =0
    # println("-- Testing cos base --")
    # println("Check sum of all components in W0: " * string(sum(mat[:,1])))
    if sum(mat[:,1]) - N >= 10e-3
        return false
    end 
    
    for i in 2:N
        # println("Check sum of all components in Wi: " * string(sum(mat[:,i])))
        if sum(mat[:,i]) >= 10e-3
            return false
        end
    end

    for i in 1:N
        for j in i+1:N
            if i != j && dot(mat[:,i], mat[:,j]) >= 10e-3
                return false 
            elseif i == j && i == 0 && dot(mat[:,i], mat[:,j]) - N >= 10e-3
                return false 
            elseif i == j && i != 0 && dot(mat[:,i], mat[:,j]) - N/2 >= 10e-3
                return false 
            end
        end
    end

    return true
end

function Get_coefficients(
        vector::Vector{Float64}, 
        ortogonal_cos_base_matrix::Matrix{Float64})

    coefficients = transpose(ortogonal_cos_base_matrix) * vector   
    
    for i in eachindex(coefficients)
        if i == 1
            coefficients[i] = coefficients[i] / (length(vector))
        else 
            coefficients[i] = coefficients[i] / (length(vector)/2)
        end
    end

    return 2 * coefficients
end

function Dct(vector::Vector{Float64})::Vector{Float64}
    base = Gen_ortogonal_cos_base(length(vector))
    return Get_coefficients(vector, base)
end

function DctII(matrix::Matrix{Float64})::Matrix{Float64}
    dct1::Matrix{Float64}

    for row in eachrow(matrix)
        vcat(dct1, Dct(row))
    end

    ris::Matrix{Float64}
    
    for col in eachcol(dct1)
        hcat(ris, Dct(col))
    end

    return ris
end

function LoadBtmImage(path::String)
    abs_path = abspath(path) 
    img = load(File{format"BMP"}(abs_path), )

    return Gray.(img)
end

function GenBtmImage()
    save("gray.bmp", colorview(Gray, rand(16,16)))
    print("fatto")
end



end