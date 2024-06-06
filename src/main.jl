include("Utils.jl")
include("Visualization.jl")
include("IterativeMethods2.jl")

using .Utils
using .Visualization
using .IterativeMethods2

using SparseArrays
using LinearAlgebra
using JSON
using Statistics

function test_methods(method::Function, A::SparseMatrixCSC, b::Vector{Float64}, tol::Float64)::Dict
    maxIter = UInt16.(20000)
    x0 = zeros(size(b))

    times = []
    memory = []
    errors = []
    iterations = []

    println("Method: ", method)
    for i in 1:10
        start = time()
        x, k = IterativeMethods2.GenericIterativeMethod(A, b, x0, tol, maxIter, method, 1.0, 1.0)
        push!(times, time() - start)
        push!(memory, (@allocated IterativeMethods2.GenericIterativeMethod(A, b, x0, tol, maxIter, method, 1.0, 1.0)) / 1e6)
        push!(errors, (norm(b - A * x) / norm(b)))
        push!(iterations, k)
    end
    println("End")

    return Dict("times" => mean(times), "std_time" => std(times),
        "memory" => mean(memory), "std_memory" => std(memory),
        "errors" => mean(errors), "std_errors" => std(errors),
        "iterations" => mean(iterations), "std_iterations" => std(iterations))
end

function test_all(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64}, tollerance::Vector{Float64})::Dict
    results = Dict()
    methods = [IterativeMethods2.JacobiMethod, IterativeMethods2.GaussSeidelMethod, IterativeMethods2.Gradient, IterativeMethods2.ConjugateGradient]

    for tol in tollerance
        results["$tol"] = Dict()
        for method in methods
            results["$tol"]["$method"] = test_methods(method, A, b, tol)
        end
    end

    return results
end

path_to_matrix = ["./data/spa1.mtx", "./data/spa2.mtx", "./data/vem1.mtx", "./data/vem2.mtx"]
tol = [1e-5, 1e-7, 1e-9, 1e-11]

total_results = Dict()

# Read the sparse matrix from the file
for path in path_to_matrix
    A = Utils.read_sparse_matrix(path)
    eig = eigvals(Matrix(A))

    println("Matrix is symmetric: ", issymmetric(A))
    println("Matrix is positive definite: ", all(eig .> 0))
    println("Condition number: ", maximum(eig) / minimum(eig))

    x = ones(size(A)[1])
    b = A * x

    total_results[String.(chop(split(path, "/")[end], tail=4))] = test_all(A, b, x, tol)
end

open("./results/results_2.json", "w") do f
    JSON.print(f, total_results)
end
