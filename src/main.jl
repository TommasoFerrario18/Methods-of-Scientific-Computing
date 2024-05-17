include("Utils.jl")
include("Analysis.jl")
include("IterativeMethods.jl")

using SparseArrays
using LinearAlgebra
using DataFrames
using CSV
using .Utils
using .Analysis
using .IterativeMethods

function run_all(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64}, xe::Vector{Float64}, tol::Vector{Float64}, file_name::String)::Tuple{DataFrame,DataFrame,DataFrame}
    maxIter = UInt16.(20000)
    x0 = zeros(size(b))

    times_df = DataFrame(Jacobi=Float64[], GaussSeidel=Float64[], Gradient=Float64[], ConjugateGradient=Float64[])
    memory_df = DataFrame(Jacobi=Float64[], GaussSeidel=Float64[], Gradient=Float64[], ConjugateGradient=Float64[])
    errors_df = DataFrame(Jacobi=Float64[], GaussSeidel=Float64[], Gradient=Float64[], ConjugateGradient=Float64[])

    for t in tol
        println("Tolerance: ", t)

        times_row = []
        memory_row = []
        errors_row = []

        print("Jacobi: \t -> \t")
        start = time()
        x, k = IterativeMethods.Jacobi(A, b, x0, t, maxIter)
        push!(times_row, time() - start)
        push!(memory_row, (@allocated IterativeMethods.Jacobi(A, b, x0, t, maxIter)) / 1e6)
        push!(errors_row, (norm(x - xe) / norm(xe)))
        if k < maxIter
            println("Iterations: ", k)
        end

        print("GaussSeidel: \t -> \t")
        start = time()
        x, k = IterativeMethods.GaussSeidel(A, b, x0, t, maxIter)
        push!(times_row, time() - start)
        push!(memory_row, (@allocated IterativeMethods.GaussSeidel(A, b, x0, t, maxIter)) / 1e6)
        push!(errors_row, (norm(x - xe) / norm(xe)))
        if k < maxIter
            println("Iterations: ", k)
        end

        print("Gradient: \t -> \t")
        start = time()
        x, k = IterativeMethods.Gradient(A, b, x0, t, maxIter)
        push!(times_row, time() - start)
        push!(memory_row, (@allocated IterativeMethods.Gradient(A, b, x0, t, maxIter)) / 1e6)
        push!(errors_row, (norm(x - xe) / norm(xe)))
        if k < maxIter
            println("Iterations: ", k)
        end

        print("ConjugateGradient: \t -> \t")
        start = time()
        x, k = IterativeMethods.ConjugateGradient(A, b, x0, t, maxIter)
        push!(times_row, time() - start)
        push!(memory_row, (@allocated IterativeMethods.ConjugateGradient(A, b, x0, t, maxIter)) / 1e6)
        push!(errors_row, (norm(x - xe) / norm(xe)))
        if k < maxIter
            println("Iterations: ", k)
        end

        push!(times_df, times_row)
        push!(memory_df, memory_row)
        push!(errors_df, errors_row)
    end

    CSV.write("./results/times_$file_name.csv", times_df)
    CSV.write("./results/memory_$file_name.csv", memory_df)
    CSV.write("./results/errors_$file_name.csv", errors_df)

    return times_df, memory_df, errors_df
end

path_to_matrix = ["./data/spa1.mtx", "./data/spa2.mtx", "./data/vem1.mtx", "./data/vem2.mtx"] 

# Read the sparse matrix from the file.
for path in path_to_matrix
    A = Utils.read_sparse_matrix(path)
    eig = eigvals(Matrix(A))

    println("Matrix is symmetric: ", issymmetric(A))
    println("Matrix is positive definite: ", all(eig .> 0))
    println("Condition number: ", maximum(eig) / minimum(eig))

    x = ones(size(A)[1])
    b = A * x

    tol = [1e-5, 1e-7, 1e-9, 1e-11]

    times_df, memory_df, errors_df = run_all(A, b, x, tol, String.(chop(split(path, "/")[end], tail=4)))

    println("Done!")
end
