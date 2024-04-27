include("Utils.jl")
include("Analysis.jl")
include("DirectMethods.jl")
include("IterativeMethods.jl")

using SparseArrays
using LinearAlgebra
using Distributions
using .Utils
using .Analysis
using .DirectMethods
using .IterativeMethods

function evaluate(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64})
    times = []
    memory = []
    errors = []

    xe = ones(size(b))
    x0 = rand(Normal(0, 1), size(b))

    N = UInt16.(size(A)[1])

    for i = 1:30
        println("Iteration: ", i)
        start = time()
        x, k = IterativeMethods.ConjugateGradient(A, b, x0, 1e-6, N)
        push!(times, time() - start)
        push!(memory, (@allocated IterativeMethods.ConjugateGradient(A, b, x0, 1e-6, N)) / 1024 / 1024)
        push!(errors, (norm(x - xe) / norm(xe)))
    end

    return times, memory, errors
end

path_to_matrix = "./data/spa1.mtx"

# Read the sparse matrix from the file.
A = Utils.read_sparse_matrix(path_to_matrix)

println("Matrix is symmetric: ", issymmetric(A))
println("Matrix is positive definite: ", all(eigvals(Matrix(A)) .> 0))

b = A * ones(size(A)[1])

times, memory, errors = evaluate(A, b)

Analysis.Visualizations(Float64.(times), Float64.(memory), Float64.(errors))

eig = eigvals(Matrix(A))
println("Condition number: ", maximum(eig) / minimum(eig))