module Dct2

using LinearAlgebra
using Plots
using FileIO
using Images
using FFTW

"""
    GenOrtogonalCosBase(dim::Integer)::Matrix{Float64}

Generate an ortogonal cos base of dimension dim x dim constructed per rows.

# Arguments
- dim::Integer: The dimension of the base.

# Returns
- Matrix{Float64}: The ortogonal cos base.
"""
function GenOrtogonalCosBase(dim::Integer)::Matrix{Float64}
    base = zeros(dim, dim)

    for k in 1:dim
        for i in 1:dim
            base[k, i] = cos((k - 1) * pi * (2 * (i - 1) + 1) / (2 * dim))
        end
    end

    return base
end

function TestOrtogonalCosBase(mat::Matrix{Float64})::Bool
    N = size(mat)[1]
    #sum =0
    # println("-- Testing cos base --")
    # println("Check sum of all components in W0: " * string(sum(mat[:,1])))
    if sum(mat[:, 1]) - N >= 10e-3
        return false
    end

    for i in 2:N
        # println("Check sum of all components in Wi: " * string(sum(mat[:,i])))
        if sum(mat[:, i]) >= 10e-3
            return false
        end
    end

    for i in 1:N
        for j in i+1:N
            if i != j && dot(mat[:, i], mat[:, j]) >= 10e-3
                return false
            elseif i == j && i == 0 && dot(mat[:, i], mat[:, j]) - N >= 10e-3
                return false
            elseif i == j && i != 0 && dot(mat[:, i], mat[:, j]) - N / 2 >= 10e-3
                return false
            end
        end
    end

    return true
end

"""
    GetCoefficients(vector::Vector{Float64}, ortogonal_cos_base_matrix::Matrix{Float64})::Vector{Float64}

Get the coefficients of the vector in the ortogonal cos base.

# Arguments
- vector::Vector{Float64}: The vector to get the coefficients.
- ortogonal_cos_base_matrix::Matrix{Float64}: The ortogonal cos base.

# Returns
- Vector{Float64}: The coefficients of the vector in the ortogonal cos base.
"""
function GetCoefficients(vector::Vector{Float64}, ortogonal_cos_base_matrix::Matrix{Float64})

    coefficients = ortogonal_cos_base_matrix * vector

    for i in eachindex(coefficients)
        if i == 1
            coefficients[i] = coefficients[i] / sqrt(length(vector))
        else
            coefficients[i] = coefficients[i] / sqrt(length(vector) / 2)
        end
    end

    return coefficients
end

"""
    Dct(vector::Vector{Float64})::Vector{Float64}

Compute the Discrete Cosine Transform of a vector.

# Arguments
- vector::Vector{Float64}: The vector to compute the DCT.

# Returns
- Vector{Float64}: The DCT coefficients of the vector.
"""
function Dct(vector::Vector{Float64})::Vector{Float64}
    base = GenOrtogonalCosBase(length(vector))
    return GetCoefficients(vector, base)
end

"""
    DctII(matrix::Matrix{Float64})::Matrix{Float64}

Compute the Discrete Cosine Transform of a matrix, applying the DCT to each row 
and column.

# Arguments
- matrix::Matrix{Float64}: The matrix to compute the DCT.

# Returns
- Matrix{Float64}: The DCT coefficients of the matrix.
"""
function DctII(matrix::Matrix{Float64})::Matrix{Float64}
    base = GenOrtogonalCosBase(size(matrix)[1])

    for i in axes(matrix, 1)
        matrix[i, :] = GetCoefficients(matrix[i, :], base)
    end

    for i in axes(matrix, 2)
        matrix[:, i] = GetCoefficients(matrix[:, i], base)
    end

    return matrix
end
"""
    DctIILibrary(matrix::Matrix{Float64})::Matrix{Float64}

Apply FFT on matrix.

# Arguments
- matrix::Matrix{Float64}: The input matrix.

# Returns
- Matrix{Float64}: The transformed matrix.
"""

function DctIILibrary(matrix::Matrix{Float64})::Matrix{Float64}
    return FFTW.plan_dct(matrix) * matrix;
end

function DctIILibrary(matrix::Matrix{UInt8})::Matrix{Float64}
    return FFTW.plan_dct(matrix) * matrix;
end

"""
    IDctIILibrary(matrix::Matrix{Float64})::Matrix{Float64}

Apply FFT on matrix.

# Arguments
- matrix::Matrix{Float64}: The input matrix.

# Returns
- Matrix{Float64}: The transformed matrix.
"""

function IDctIILibrary(matrix::Matrix{Float64})::Matrix{Float64}
    return FFTW.plan_idct(matrix) * matrix;
end


"""
    ResizeMatrix(img::Matrix{UInt8}, F::Int64)::Matrix{UInt8}

Resize the image to be a multiple of F.

# Arguments
- img::Matrix{UInt8}: The image to resize.
- F::Int64: The factor to resize the image.

# Returns
- Matrix{UInt8}: The resized image.
"""
function ResizeMatrix(img::Matrix{UInt8}, F::Int64)::Matrix{UInt8}
    return img[1:size(img)[1]-(size(img)[1]%F), 1:size(img)[2]-(size(img)[2]%F)]
end

"""
    Compress(c::Matrix{Float64}, d::Int64)::Matrix{Float64}

Compress the DCT coefficients by setting to 0 the coefficients with index i + j - 2 >= d.

# Arguments
- c::Matrix{Float64}: The DCT coefficients.
- d::Int64: The threshold to compress the coefficients.

# Returns
- Matrix{Float64}: The compressed DCT coefficients.
"""
function Compress(c::Matrix{Float64}, d::Int64)::Matrix{Float64}
    for i in axes(c, 1)
        for j in axes(c, 2)
            if i + j - 2 >= d
                c[i, j] = 0
            end
        end
    end
    return c
end

"""
    Normalize(c::Matrix{Float64})::Matrix{UInt8}

Normalize the DCT coefficients to be in the range [0, 255].

# Arguments
- c::Matrix{Float64}: The DCT coefficients.

# Returns
- Matrix{UInt8}: The normalized DCT coefficients.
"""
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

"""
    ApplyDct2OnImage(img::Matrix{UInt8}, F::Int64, d::Int64)::Matrix{UInt8}

Apply the DCT to the image by blocks of size F x F and compress the coefficients with a threshold d.

# Arguments
- img::Matrix{UInt8}: The image to apply the DCT.
- F::Int64: The size of the blocks.
- d::Int64: The threshold to compress the coefficients.

# Returns
- Matrix{UInt8}: The image with the DCT applied.
"""
function ApplyDct2OnImage(img::Matrix{UInt8}, F::Int64, d::Int64)::Matrix{UInt8}
    img = ResizeMatrix(img, F)

    c = zero(rand(F, F))

    for i in 0:div(size(img)[1], F)-1
        for j in 0:div(size(img)[2], F)-1
            c = img[i*F+1:i*F+F, j*F+1:j*F+F]
            c = Dct2.DctIILibrary(c)
            c = Dct2.Compress(c, d)
            c = Dct2.IDctIILibrary(c)
            c = Dct2.Normalize(c)
            img[i*F+1:i*F+F, j*F+1:j*F+F] = c
        end
    end

    return img
end

end