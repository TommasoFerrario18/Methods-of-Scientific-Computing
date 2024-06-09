include("Utils.jl")
include("Visualization.jl")
include("IterativeMethods.jl")

using .Utils
using .Visualization
using .IterativeMethods

using SparseArrays
using LinearAlgebra
using JSON
using Statistics
using IterativeSolvers

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
        x, k = IterativeMethods.GenericIterativeMethod(A, b, x0, tol, maxIter, method, 1.0, 1.0)
        push!(times, time() - start)
        push!(memory, (@allocated IterativeMethods.GenericIterativeMethod(A, b, x0, tol, maxIter, method, 1.0, 1.0)) / 1e6)
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
    # methods = [IterativeMethods.JacobiMethod, IterativeMethods.GaussSeidelMethod, IterativeMethods.Gradient, IterativeMethods.ConjugateGradient]
    methods = [IterativeMethods.GaussSeidelMethod]
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
    println("Matrix: ", path)
    println("Matrix is symmetric: ", issymmetric(A))
    println("Matrix is positive definite: ", all(eig .> 0))
    println("Condition number: ", maximum(eig) / minimum(eig))
    println("Matrix is diagonally dominant: ", Utils.is_diagonally_dominant(Matrix(A)))

    x = ones(size(A)[1])
    b = A * x

    total_results[String.(chop(split(path, "/")[end], tail=4))] = test_all(A, b, tol)
end

open("./results/results_gauss.json", "w") do f
    JSON.print(f, total_results)
end

# A = Utils.read_sparse_matrix("./data/vem1.mtx")
# xe = ones(size(A)[1])
# b = A * xe
# x = zeros(size(b))
# println("GaussSeidelMethod")
# start = time()
# gauss_seidel!(x, A::SparseMatrixCSC, b; maxiter=10)
# println(time() - start, "\nGaussSeidelMethod")
# start = time()
# my_x, k = IterativeMethods.GenericIterativeMethod(A, b, zeros(size(b)), 1e-100, UInt16.(10), IterativeMethods.GaussSeidelMethod, 1.0, 1.0)
# println(time() - start)

# println("My x: ", norm(b - A * my_x) / norm(b))
# println("x: ", norm(b - A * x) / norm(b))