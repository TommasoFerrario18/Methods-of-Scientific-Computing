
module Dct2

using LinearAlgebra



function Gen_ortogonal_cos_base(dim::Integer)::Matrix{Float64}
    base = zeros(dim,dim)
    

    for i in 1:dim
        for k in 1:dim
            base[i, k] = cos(k * pi * (2 * (i - 1) + 1) / (2 * dim))
        end
    end

    return base
end

function Test_ortogonal_cos_base(mat::Matrix{Float64})::Bool
    N = size(mat)[1]
    #sum =0
    if sum(mat[:,1]) != N
        print("F")
        return false
    end 
    
    for i in 2:N
        sum = 0
        for j in 1:N
            sum += mat[j,i]
        end
        if sum != 0
            return false
        end
    end

    for i in 1:N
        for j in i+1:N
            if i != j & dot(mat[:,i], mat[:,j]) != 0
                return false 
            elseif i == j & i == 0 & dot(mat[:,i], mat[:,j]) != N
                return false 
            elseif i == j & i != 0 & dot(mat[:,i], mat[:,j]) != N/2
                return false 
            end
        end
    end

    return true
end 


function Map_vector_from_canonic_base_to_ortogonal_cos_base(
        vector::Vector{Float64}, 
        ortogonal_cos_base_matrix::Matrix{Float64})
    coefficients =  collect(transpose(vector)) * ortogonal_cos_base_matrix   
    for (index, coef) in enumerate(coefficients)
        if index == 1
            coef /= length(vector)
        else 
            coef /= length(vector)/2
        end
    end
    """
    println(coefficients)
    print(typeof(coefficients))
    println(ortogonal_cos_base_matrix)
    print(typeof(ortogonal_cos_base_matrix))
    """
    return coefficients * ortogonal_cos_base_matrix 
end

function Dct(vector::Vector{Float64})
    base = Gen_ortogonal_cos_base(length(vector))
    println("Check BASE ")
    println(Test_ortogonal_cos_base(base))
    println("base")
    println(base)
    vector_cos_base = Map_vector_from_canonic_base_to_ortogonal_cos_base(vector,base)
    return vector_cos_base
end
"""

function Dct2(matrix::Matrix{Float64})::Matrix{Float64}

end
"""


end