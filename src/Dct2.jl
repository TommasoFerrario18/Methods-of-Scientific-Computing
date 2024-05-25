module Dct2

include("Utils.jl")

using LinearAlgebra
using Plots
using FileIO
using Images
using FFTW
using .Utils
using Distributed



function GenOrtogonalCosBase(dim::Integer)::Matrix{Float64}
    base = zeros(dim,dim) 

    for k in 1:dim 
        for i in 1:dim
            base[k, i] = cos((k-1) * pi * (2 * (i-1) + 1) / (2 * dim))
        end
    end

    return base
end

function TestOrtogonalCosBase(mat::Matrix{Float64})::Bool
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

function GetCoefficients(
        vector::Vector{Float64}, 
        ortogonal_cos_base_matrix::Matrix{Float64})

    coefficients = ortogonal_cos_base_matrix * vector   

    for i in eachindex(coefficients)
        if i == 1
            coefficients[i] = coefficients[i] / sqrt(length(vector))
        else 
            coefficients[i] = coefficients[i] / sqrt(length(vector)/2)
        end
    end

    return coefficients
end

function Dct(vector::Vector{Float64})::Vector{Float64}
    base = GenOrtogonalCosBase(length(vector))
    return GetCoefficients(vector, base)
end

function DctII(matrix::Matrix{Float64})::Matrix{Float64}
    base = GenOrtogonalCosBase(size(matrix)[1])

    for i in axes(matrix, 1) 
        matrix[i,:] = GetCoefficients(matrix[i,:], base)
    end

    for i in axes(matrix, 2) 
        matrix[:,i] = GetCoefficients(matrix[:,i], base)
    end

    return matrix
end

function ResizeMatrix(img::Matrix{UInt8}, F::Int64)::Matrix{UInt8}
    return img[1:size(img)[1] - (size(img)[1]% F), 1:size(img)[2] - (size(img)[2]% F)]
end

function Compress(c::Matrix{Float64}, d::Int64)::Matrix{Float64}
    for i in axes(c,1)
        for j in axes(c,2)
            if i + j - 2 >= d
                c[i, j] = 0
            end
        end
    end
    return c
end

function Normalize(c::Matrix{Float64})::Matrix{UInt8}
    c = round.(c)
    for i in eachindex(c)
        if c[i] > 255
            c[i] = 255
        elseif c[i] < 0
            c[i] = 0
        end
    end
    return c
end

function ApplyDct2OnImage(img::Matrix{UInt8}, F::Int64, d::Int64)::Matrix{UInt8}
    img = ResizeMatrix(img, F)

    c = zero(rand(F,F))

    for i in 0:div(size(img)[1], F)-1
        for j in 0:div(size(img)[2], F)-1
            c = img[i * F + 1 : i * F + F, j * F + 1 : j * F + F]
            c = FFTW.dct(c)
            c = Dct2.Compress(c, d)
            c = FFTW.idct(c)
            c = Dct2.Normalize(c)
            img[i * F + 1 : i * F + F, j * F + 1 : j * F + F] = c
        end
    end
    
    return img
end

end