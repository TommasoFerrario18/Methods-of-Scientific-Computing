module IterativeMethods2
include("Utils.jl")
include("DirectMethods.jl")

using .Utils
using .DirectMethods
using LinearAlgebra
using SparseArrays

"""
    RemainingStoppingCriteria(A, x, b, tol)

Check if the stopping criteria for an iterative method has been reached.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `x::Vector{Float64}`: The current solution vector.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `tol::Float64`: The tolerance value.

# Returns
- `Bool`: `true` if the stopping criteria has been reached, `false` otherwise.

"""
function RemainingStoppingCriteria(A::SparseMatrixCSC{Float64,UInt32},
    x::Vector{Float64}, b::Vector{Float64}, tol::Float64)::Bool
    return RemainingStoppingCriteria((b - (A * x)), b, tol)
end

"""
    RemainingStoppingCriteria(r::Vector{Float64}, b::Vector{Float64}, tol::Float64)

Check if the stopping criteria for an iterative method has been reached.

# Arguments
- `r::Vector{Float64}`: The residual vector.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `tol::Float64`: The tolerance value.

# Returns
- `Bool`: `true` if the stopping criteria has been reached, `false` otherwise.

"""
function RemainingStoppingCriteria(r::Vector{Float64}, b::Vector{Float64}, tol::Float64)::Bool
    return norm(r) / norm(b) < tol
end

"""
    GenericIterativeMethod(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    x0::Vector{Float64}, tol::Float64, max_iter::UInt16, method::Function, w::Float64, alpha::Float64)

Solve a linear system using an iterative method.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial guess for the solution.
- `tol::Float64`: The tolerance value.
- `max_iter::UInt16`: The maximum number of iterations.
- `method::Function`: The iterative method to use.
- `w::Float64`: The relaxation factor.
- `alpha::Float64`: The parameter for the SOR method.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function GenericIterativeMethod(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    x0::Vector{Float64}, tol::Float64, max_iter::UInt16, method::Function, w::Float64, alpha::Float64)::Tuple{Vector{Float64},UInt16}
    n = size(A)[1]
    number_of_iterations = UInt16.(0)

    if n != size(A)[2] || n != size(b)[1] || n != size(x0)[1]
        throw(ArgumentError("The dimensions of the input matrices are not compatible"))
    end

    x = copy(x0)
    r = b - (A * x)
    while !RemainingStoppingCriteria(r, b, tol)
        x = method(A, b, x, w, alpha)
        r = b - (A * x)
        number_of_iterations += 1
        if number_of_iterations >= max_iter
            println("The method did not converge")
            return x, number_of_iterations
        end
    end

    return x, number_of_iterations
end

end