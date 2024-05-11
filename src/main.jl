include("Utils.jl")
include("Analysis.jl")
include("DirectMethods.jl")
include("IterativeMethods.jl")

using SparseArrays
using LinearAlgebra
using DataFrames
using .Utils
using .Analysis
using .DirectMethods
using .IterativeMethods

function evaluate(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64})
    times = []
    memory = []
    errors = []

    xe = ones(size(b))
    x0 = zeros(size(b))
    N = UInt16.(size(A)[1])

    for i = 1:30
        println("Iteration: ", i)
        start = time()
        x, k = IterativeMethods.ConjugateGradient(A, b, x0, 1e-6, N)
        push!(times, time() - start)
        push!(memory, (@allocated IterativeMethods.ConjugateGradient(A, b, x0, 1e-6, N)) / 1e6)
        push!(errors, (norm(x - xe) / norm(xe)))
    end

    return times, memory, errors
end

function run_all(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64}, x::Vector{Float64}, tol::Vector{Float64})::Tuple{DataFrame,DataFrame,DataFrame}
    maxIter = UInt16.(20000)
    x0 = zeros(size(b))
    xe = ones(size(b))

    times_df = DataFrame(Jacobi=Float64[], GaussSeidel=Float64[], Gradient=Float64[], ConjugateGradient=Float64[])
    memory_df = DataFrame(Jacobi=Float64[], GaussSeidel=Float64[], Gradient=Float64[], ConjugateGradient=Float64[])
    errors_df = DataFrame(Jacobi=Float64[], GaussSeidel=Float64[], Gradient=Float64[], ConjugateGradient=Float64[])

    for t in tol
        println("Tolerance: ", t)

        times_row = []
        memory_row = []
        errors_row = []

        println("Jacobi")
        start = time()
        x, k = IterativeMethods.Jacobi(A, b, x0, t, maxIter)
        push!(times_row, time() - start)
        push!(memory_row, (@allocated IterativeMethods.Jacobi(A, b, x0, t, maxIter)) / 1e6)
        push!(errors_row, (norm(x - xe) / norm(xe)))

        println("GaussSeidel")
        start = time()
        x, k = IterativeMethods.GaussSeidel(A, b, x0, t, maxIter)
        push!(times_row, time() - start)
        push!(memory_row, (@allocated IterativeMethods.GaussSeidel(A, b, x0, t, maxIter)) / 1e6)
        push!(errors_row, (norm(x - xe) / norm(xe)))

        println("Gradient")
        start = time()
        x, k = IterativeMethods.Gradient(A, b, x0, t, maxIter)
        push!(times_row, time() - start)
        push!(memory_row, (@allocated IterativeMethods.Gradient(A, b, x0, t, maxIter)) / 1e6)
        push!(errors_row, (norm(x - xe) / norm(xe)))

        println("ConjugateGradient")
        start = time()
        x, k = IterativeMethods.ConjugateGradient(A, b, x0, t, maxIter)
        push!(times_row, time() - start)
        push!(memory_row, (@allocated IterativeMethods.ConjugateGradient(A, b, x0, t, maxIter)) / 1e6)
        push!(errors_row, (norm(x - xe) / norm(xe)))

        push!(times_df, times_row)
        push!(memory_df, memory_row)
        push!(errors_df, errors_row)
    end

    return times_df, memory_df, errors_df
end



path_to_matrix = "./data/spa2.mtx"

# Read the sparse matrix from the file.
A = Utils.read_sparse_matrix(path_to_matrix)
eig = eigvals(Matrix(A))

println("Matrix is symmetric: ", issymmetric(A))
println("Matrix is positive definite: ", all(eig .> 0))
println("Condition number: ", maximum(eig) / minimum(eig))

x = ones(size(A)[1])
b = A * x

tol = [1e-5, 1e-7, 1e-9, 1e-11]

times_df, memory_df, errors_df = run_all(A, b, x, tol)

println("Times")
println(times_df)
println("Memory")
println(memory_df)
println("Errors")
println(errors_df)