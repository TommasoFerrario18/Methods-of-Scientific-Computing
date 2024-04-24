module DirectMethods

using LinearAlgebra
using SparseArrays

"""
    BackwardSubstitution(A, b)

Solves the linear system `Ax = b` using backward substitution.

# Arguments
- `A`: The coefficient matrix of size `n x n`.
- `b`: The right-hand side vector of size `n`.

# Returns
- The solution vector `x` of size `n`.

# Errors
- Throws an error if `A` is not square.
- Throws an error if `A` and `b` do not have the same number of rows.
- Throws an error if `A` has a zero in the diagonal.

"""
function BackwardSubstitution(A::SparseMatrixCSC{Float64,Int64}, b::Vector{Float64})::Vector{Float64}
    n = size(A)[1]
    x = zeros(n)

    check_sizes(A, b)

    if A[n, n] == 0
        error("Matrix A have a zero in the diagonal")
    end

    x[n] = b[n] / A[n, n]

    for i = n-1:-1:1
        if A[i, i] == 0
            error("Matrix A have a zero in the diagonal")
        end

        x[i] = (b[i] - (dot(A[i, :], x))) / A[i, i]
    end

    return x
end

"""
    ForwardSubstitution(A, b)

Solves the linear system `Ax = b` using forward substitution.

# Arguments
- `A`: The coefficient matrix of size `n x n`.
- `b`: The right-hand side vector of size `n`.

# Returns
- The solution vector `x` of size `n`.

# Errors
- Throws an error if `A` is not square.
- Throws an error if `A` and `b` do not have the same number of rows.
- Throws an error if `A` has a zero in the diagonal.

"""
function ForwardSubstitution(A::SparseMatrixCSC{Float64, Int64}, b::Vector{<:Real})::Vector{<:Real}
    n = size(A)[1]
    x = zeros(n)

    check_sizes(A, b)

    if A[1, 1] == 0
        error("Matrix A have a zero in the diagonal")
    end

    x[1] = b[1] / A[1, 1]

    for i = 2:n
        if A[i, i] == 0
            error("Matrix A have a zero in the diagonal")
        end

        x[i] = (b[i] - (dot(A[i, :], x))) / A[i, i]
    end

    return x
end

function check_sizes(A::SparseMatrixCSC{Float64,Int64}, b::Vector{Float64})
    if size(A)[1] != size(b)[1] || size(A)[2] != size(A)[1]
        error("Matrix A and vector b must have the same number of rows")
    end
end


"""
    CholeskyDecomposition(A::SparseMatrixCSC{Float64,Int64})::SparseMatrixCSC{Float64,Int64}

Perform Cholesky decomposition on a given sparse matrix `A` and return the 
resulting lower triangular matrix.

# Arguments
- `A::SparseMatrixCSC{Float64,Int64}`: The input sparse matrix.

# Returns
- `SparseMatrixCSC{Float64,Int64}`: The lower triangular matrix resulting from 
the Cholesky decomposition.

"""
function CholeskyDecomposition(A::SparseMatrixCSC{Float64,Int64})::SparseMatrixCSC{Float64,Int64}
    n = size(A)[1]
    if n != size(A)[2]
        error("Matrix A must be square")
    end

    rows_index = Int64[]
    cols_index = Int64[]
    values = Float64[]

    R = zeros(n, n)

    A_star = copy(A)

    for k = 1:n
        rkk = sqrt(A_star[k, k])
        push!(rows_index, k)
        push!(cols_index, k)
        push!(values, rkk)

        for j = k+1:n
            rkj = A_star[k, j] / rkk
            push!(rows_index, k)
            push!(cols_index, j)
            push!(values, rkj)
        end

        rho = 1 / A_star[k, k]

        for j = k+1:n
            for i = k+1:n
                A_star[i, j] = A_star[i, j] - rho * A_star[i, k] * A_star[k, j]
            end
        end
    end

    return sparse(rows_index, cols_index, values)
end

"""
    Cholesky(A::SparseMatrixCSC{Float64,Int64}, b::Vector{<:Real})::Vector{<:Real}

Solves the linear system `Ax = b` using the Cholesky decomposition method.

# Arguments
- `A`: The input sparse matrix `A` of type `SparseMatrixCSC{Float64,Int64}`.
- `b`: The input vector `b` of type `Vector{<:Real}`.

# Returns
- The solution vector `x` of type `Vector{<:Real}`.

"""
function Cholesky(A::SparseMatrixCSC{Float64,Int64}, b::Vector{<:Real})::Vector{<:Real}
    R = CholeskyDecomposition(A)
    y = ForwardSubstitution(Float64.(conj(R')), b)
    return BackwardSubstitution(R, y)
end

end