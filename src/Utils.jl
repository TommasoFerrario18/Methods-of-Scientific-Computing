module Utils

using SparseArrays
using Plots
using Statistics

"""
    read_sparse_matrix(file_path::String)::SparseMatrixCSC{Float64,Int64}

Reads a sparse matrix from a file.

# Arguments
- `file_path::String`: The path to the file containing the sparse matrix.

# Returns
- `SparseMatrixCSC{Float64,Int64}`: The sparse matrix read from the file.
"""
function read_sparse_matrix(file_path::String)::SparseMatrixCSC{Float64,Int64}
    rows_index = Int64[]
    cols_index = Int64[]
    values = Float64[]

    try
        file = open(file_path, "r")

        for line in Iterators.drop(eachline(file), 1)
            if line[1] == '%'
                continue
            end
            row, col, value = split(line)
            push!(rows_index, parse(Int64, row))
            push!(cols_index, parse(Int64, col))
            push!(values, parse(Float64, value))
        end

        close(file)
    catch e
        println("Error: Could not open file.")
    end

    return sparse(rows_index, cols_index, values)
end

"""
    plot_results(times, memory, errors)

Plot the results of a Cholesky decomposition algorithm.

# Arguments
- `times`: An array of time values for each iteration.
- `memory`: An array of memory usage values for each iteration.
- `errors`: An array of error values for each iteration.

# Returns
- Nothing

"""
function plot_results(times, memory, errors)
    println("Statistics| mean | std | min | max")
    println("Time: ", round(mean(times), digits=3), " s", " | ", round(std(times),
            digits=3), " s", " | ", round(minimum(times), digits=3), " s", " | ",
        round(maximum(times), digits=3), " s")
    println("Memory: ", round(mean(memory), digits=3), " bytes", " | ",
        round(std(memory), digits=3), " bytes", " | ", round(minimum(memory),
            digits=3), " bytes", " | ", round(maximum(memory), digits=3), " bytes")
    println("Error: ", round(mean(errors), digits=3), " | ", round(std(errors),
            digits=3), " | ", round(minimum(errors), digits=3), " | ",
        round(maximum(errors), digits=3))

    p1 = plot(times, label="Time", title="Cholesky Decomposition", xlabel="Iteration",
        ylabel="Time (s)", linewidth=2, legend=:topleft)
    p2 = plot(memory, label="Memory", title="Cholesky Decomposition", xlabel="Iteration", ylabel="Memory (bytes)")
    p3 = plot(errors, label="Error", title="Cholesky Decomposition", xlabel="Iteration", ylabel="Error")

    plot(p1, p2, p3)
end
end