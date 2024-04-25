include("Utils.jl")
include("Analysis.jl")
include("DirectMethods.jl")
include("IterativeMethods.jl")

using SparseArrays
using LinearAlgebra
using .Utils
using .Analysis
using .DirectMethods
using .IterativeMethods

function evaluate(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64})
    times = []
    memory = []
    errors = []

    xe = ones(size(b))

    N = 5000

    for i = 1:10
        println("Iteration: ", i)
        x, k = IterativeMethods.Gradient(A, b, zeros(size(b)), 1e-6, UInt16.(N))
        push!(times, @elapsed IterativeMethods.Gradient(A, b, zeros(size(b)), 1e-6, UInt16.(N)))
        push!(memory, @allocated IterativeMethods.Gradient(A, b, zeros(size(b)), 1e-6, UInt16.(N)))
        push!(errors, (norm(x - xe) / norm(xe)))
    end

    return times, memory, errors
end

path_to_matrix = "./data/spa1.mtx"

# Read the sparse matrix from the file.
A = Utils.read_sparse_matrix(path_to_matrix)
b = A * ones(size(A)[1])

times, memory, errors = evaluate(A, b)

Analysis.Visualizations(Float64.(times), Int64.(memory), Float64.(errors))
