module DirectMethods

include("Utils.jl")

using LinearAlgebra
using SparseArrays
using .Utils

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
function BackwardSubstitution(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64})::Vector{Float64}
    n = size(A)[1]
    x = zeros(n)

    Utils.check_sizes(A, b)

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
function ForwardSubstitution(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{<:Real})::Vector{<:Real}
    n = size(A)[1]
    x = similar(b)

    Utils.check_sizes(A, b)

    if A[1, 1] == 0
        error("Matrix A have a zero in the diagonal")
    end

    x[1] = b[1] / A[1, 1]

    for i = 2:n
        diag = A[i, i]
        if diag == 0
            error("Matrix A have a zero in the diagonal")
        end

        x[i] = (b[i] - (dot(A[i, :], x))) / diag
    end

    return x
end

"""
    GaussReduction(A, b, pivot="partial")

Solves a system of linear equations using Gaussian elimination with optional pivoting.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The coefficient matrix of the system.
- `b::Vector{Float64}`: The right-hand side vector of the system.
- `pivot::String="partial"`: The type of pivoting to use. Options are "partial" (partial pivoting) or "total" (total pivoting).

# Returns
- `x::Vector{Float64}`: The solution vector of the system.

"""
function GaussReduction(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    pivot::String="partial")::Vector{Float64}
    n = size(A)[1]

    Utils.check_sizes(A, b)

    U = copy(A)
    new_b = copy(b)

    columns_swap = UInt32[]

    for k = 1:n-1
        pivot_matrix = U[k:n, k:n]

        if pivot == "partial"
            s = Utils.PartialPivot(pivot_matrix)
            Utils.swapRow(U, k, (k - 1 + s))
            Utils.swapVectorPosition(new_b, k, (k - 1 + s))
        else
            row, col = Utils.TotalPivot(pivot_matrix)
            # Scambio le righe -> scambiare i termini noti anche
            Utils.swapRow(U, k, (k - 1 + row))
            Utils.swapVectorPosition(new_b, k, (k - 1 + row))
            # Scambio le colonne -> scambiare le incognite
            Utils.swapColumn(U, k, (k - 1 + col))
            push!(columns_swap, (k, (k - 1 + col)))
        end

        for i = k+1:n
            m = U[i, k] / U[k, k]

            U[i, :] = U[i, :] - (m .* U[k, :])

            new_b[i] = new_b[i] - m * new_b[k]
        end
    end

    x = BackwardSubstitution(U, new_b)

    for (i, j) in columns_swap
        temp = x[i]
        x[i] = x[j]
        x[j] = temp
    end

    return x
end

"""
    PALUDecomposition(A::SparseMatrixCSC{Float64,UInt32}) -> Tuple{SparseMatrixCSC{Float64,UInt32},SparseMatrixCSC{Float64,UInt32},SparseMatrixCSC{Float64,UInt32}}

Perform the PALU decomposition of a square sparse matrix A.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The input sparse matrix.

# Returns
- `Tuple{SparseMatrixCSC{Float64,UInt32},SparseMatrixCSC{Float64,UInt32},SparseMatrixCSC{Float64,UInt32}}`: A tuple containing the lower triangular matrix L, the upper triangular matrix U, and the permutation matrix P.

"""
function PALUDecomposition(A::SparseMatrixCSC{Float64,UInt32}, pivoting::String="partial")::Tuple{SparseMatrixCSC{Float64,UInt32},SparseMatrixCSC{Float64,UInt32},SparseMatrixCSC{Float64,UInt32}}
    n = size(A)[1]
    if n != size(A)[2]
        error("Matrix A must be square")
    end

    U = copy(A)
    L = spzeros(n, n)
    P = sparse(I, n, n) # Serve per tenere traccia delle permutazioni

    for k = 1:n-1
        pivot_matrix = U[k:n, k:n]
        if pivoting == "partial"
            s = Utils.PartialPivot(pivot_matrix)

            Utils.swapRow(U, k, (k - 1 + s))
            Utils.swapRow(L, k, (k - 1 + s))
            Utils.swapRow(P, k, (k - 1 + s))
        elseif pivoting == "total"
            row, col = Utils.TotalPivot(pivot_matrix)

            Utils.swapRow(U, k, (k - 1 + row))
            Utils.swapRow(L, k, (k - 1 + row))
            Utils.swapRow(P, k, (k - 1 + row))

            Utils.swapColumn(U, k, (k - 1 + col))
            Utils.swapColumn(L, k, (k - 1 + col))
            Utils.swapColumn(P, k, (k - 1 + col))
        end

        for i = k+1:n
            m = U[i, k] / U[k, k]
            L[i, k] = m
            U[i, :] = U[i, :] - m * U[k, :]
        end
    end

    L = L + I
    return L, U, P
end

"""
    PALU(A::SparseMatrixCSC{Float64,UInt32}, b::Matrix{Float64})::Matrix{Float64}

Performs the PALU decomposition and solves the linear system `Ax = b` using forward and backward substitution.

# Arguments
- `A`: The coefficient matrix `A` of the linear system, represented as a sparse matrix in Compressed Sparse Column (CSC) format.
- `b`: The right-hand side vector `b` of the linear system.

# Returns
- The solution vector `x` of the linear system.

"""
function PALU(A::SparseMatrixCSC{Float64,UInt32}, b::Matrix{Float64})::Matrix{Float64}
    if size(b)[1] != size(A)[1]
        error("Matrix A and vector b must have the same number of rows")
    end

    L, U, P = PALUDecomposition(A)

    res = zeros(size(b))

    for i = 1:size(b)[2]
        y = ForwardSubstitution(L, P * b[:, i])
        x = BackwardSubstitution(U, y)
        res[:, i] = x
    end

    return res
end

"""
    CholeskyDecomposition(A::SparseMatrixCSC{Float64,UInt32})::SparseMatrixCSC{Float64,UInt32}

Perform Cholesky decomposition on a given sparse matrix `A` and return the 
resulting lower triangular matrix.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The input sparse matrix.

# Returns
- `SparseMatrixCSC{Float64,UInt32}`: The lower triangular matrix resulting from 
the Cholesky decomposition.

"""
function CholeskyDecomposition(A::SparseMatrixCSC{Float64,UInt32})::SparseMatrixCSC{Float64,UInt32}
    n = size(A)[1]
    if n != size(A)[2]
        error("Matrix A must be square")
    end

    rows_index = UInt32[]
    cols_index = UInt32[]
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
    Cholesky(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{<:Real})::Vector{<:Real}

Solves the linear system `Ax = b` using the Cholesky decomposition method.

# Arguments
- `A`: The input sparse matrix `A` of type `SparseMatrixCSC{Float64,UInt32}`.
- `b`: The input vector `b` of type `Vector{<:Real}`.

# Returns
- The solution vector `x` of type `Vector{<:Real}`.

"""
function Cholesky(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{<:Real})::Vector{<:Real}
    R = CholeskyDecomposition(A)
    y = ForwardSubstitution(Float64.(conj(R')), b)
    return BackwardSubstitution(R, y)
end

end