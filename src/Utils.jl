module Utils

using SparseArrays
using Plots
using Statistics

"""
    read_sparse_matrix(file_path::String)::SparseMatrixCSC{Float64,UInt32}

Reads a sparse matrix from a file.

# Arguments
- `file_path::String`: The path to the file containing the sparse matrix.

# Returns
- `SparseMatrixCSC{Float64,UInt32}`: The sparse matrix read from the file.
"""
function read_sparse_matrix(file_path::String)::SparseMatrixCSC{Float64,UInt32}
    rows_index = UInt32[]
    cols_index = UInt32[]
    values = Float64[]

    try
        file = open(file_path, "r")

        for line in Iterators.drop(eachline(file), 1)
            if line[1] == '%'
                continue
            end
            row, col, value = split(line)
            push!(rows_index, parse(UInt32, row))
            push!(cols_index, parse(UInt32, col))
            push!(values, parse(Float64, value))
        end

        close(file)
    catch e
        println("Error: Could not open file.")
    end

    return sparse(rows_index, cols_index, values)
end

"""
    PartialPivot(A)

Finds the row with the largest absolute value in the first column of matrix `A`.

# Arguments
- `A`: The matrix of size `n x n`.

# Returns
- The row with the largest absolute value in the first column of `A`.

"""
function PartialPivot(A::SparseMatrixCSC{Float64,UInt32})::UInt32
    n = size(A)[1]
    max = 0
    s = 0

    for i = 1:n
        if abs(A[i, 1]) > max
            max = abs(A[i, 1])
            s = i
        end
    end

    return s
end

"""
    TotalPivot(A)

Finds the row with the largest absolute value in matrix `A`.

# Arguments
- `A`: The matrix of size `n x n`.

# Returns
- The row with the largest absolute value in `A`.

"""
function TotalPivot(A::SparseMatrixCSC{Float64,UInt32})
    n = size(A)[1]
    max = 0
    row = 0
    col = 0

    for i = 1:n
        for j = 1:n
            if abs(A[i, j]) > max
                max = abs(A[i, j])
                row = i
                col = j
            end
        end
    end

    return row, col
end

"""
    swapRow(A, i, j)

Swaps rows `i` and `j` in matrix `A`.

# Arguments
- `A`: The matrix of size `n x n`.
- `i`: The index of the first row to swap.
- `j`: The index of the second row to swap.

# Errors
- Throws an error if `i` or `j` is out of bounds.

"""
function swapRow(A::SparseMatrixCSC{Float64,UInt32}, i::Integer, j::Integer)
    n = size(A)[1]

    if i > n || j > n || i < 1 || j < 1
        error("Index out of bounds")
    end

    temp = copy(A[i, :])
    A[i, :] = A[j, :]
    A[j, :] = temp
end

"""
    swapColumn(A, i, j)

Swaps columns `i` and `j` in matrix `A`.

# Arguments
- `A`: The matrix of size `n x n`.
- `i`: The index of the first column to swap.
- `j`: The index of the second column to swap.

# Errors
- Throws an error if `i` or `j` is out of bounds.
"""
function swapColumn(A::SparseMatrixCSC{Float64,UInt32}, i::Integer, j::Integer)
    n = size(A)[1]

    if i > n || j > n || i < 1 || j < 1
        error("Index out of bounds")
    end

    temp = copy(A[:, i])
    A[:, i] = A[:, j]
    A[:, j] = temp
end

"""
    swapVectorPosition(x, i, j)

Swaps elements `i` and `j` in vector `x`.

# Arguments
- `x`: The vector of size `n`.
- `i`: The index of the first element to swap.
- `j`: The index of the second element to swap.

# Errors
- Throws an error if `i` or `j` is out of bounds.

"""
function swapVectorPosition(x::Vector{<:Real}, i::Integer, j::Integer)
    n = size(x)[1]

    if i > n || j > n || i < 1 || j < 1
        error("Index out of bounds")
    end

    temp = x[i]
    x[i] = x[j]
    x[j] = temp
end

"""
    check_sizes(A, b)

Checks if the matrix `A` and vector `b` have the same number of rows.

# Arguments
- `A`: The matrix of size `n x n`.
- `b`: The vector of size `n`.

# Errors
- Throws an error if `A` and `b` do not have the same number of rows.

"""
function check_sizes(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64})
    if size(A)[1] != size(b)[1] || size(A)[2] != size(A)[1]
        error("Matrix A and vector b must have the same number of rows")
    end
end

"""
    optimal_alpha_richardson(P::SparseMatrixCSC{Float64,UInt32})::Float64

Computes the optimal value of the relaxation parameter `alpha` for the Richardson method.

# Arguments
- `P`: The preconditioner matrix.

# Returns
- The optimal value of the relaxation parameter `alpha`.

"""
function optimal_alpha_richardson(P::SparseMatrixCSC{Float64,UInt32})::Float64
    max_eg = maximum(eigvals(Matrix(P)))
    min_eg = minimum(eigvals(Matrix(P)))

    return 2 / (max_eg + min_eg)
end
end