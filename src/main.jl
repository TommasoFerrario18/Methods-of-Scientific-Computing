include("Utils.jl")
include("DirectMethods.jl")

using SparseArrays
using LinearAlgebra
using .Utils
using .DirectMethods

function evaluate(A::SparseMatrixCSC{Float64,Int64}, b::Vector{Float64})
    times = []
    memory = []
    errors = []

    xe = ones(size(b))

    for i = 1:10
        x = DirectMethods.Cholesky(A, b)
        push!(times, @elapsed DirectMethods.Cholesky(A, b))
        push!(memory, @allocated DirectMethods.Cholesky(A, b))
        push!(errors, (norm(x - xe) / norm(xe)))
    end

    return times, memory, errors
end

path_to_matrix = "./data/spa1.mtx"

# Read the sparse matrix from the file.
A = Utils.read_sparse_matrix(path_to_matrix)
b = A * ones(size(A)[1])

times, memory, errors = evaluate(A, b)

Utils.plot_results(times, memory, errors)
