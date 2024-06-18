module IterativeMethods
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

    if method == IterativeMethods.JacobiMethod
        if w <= 0 || w > 1
            throw(ArgumentError("The relaxation factor must be in the interval (0, 1]"))
        end

        P = spdiagm(0 => w ./ diag(A))
    elseif method == IterativeMethods.GaussSeidelMethod
        if w <= 0 || w >= 2
            throw(ArgumentError("The relaxation factor must be in the interval (0, 2)"))
        end
        P = sparse(tril(A))

        for i = 1:n
            P[i, i] = P[i, i] / w
        end
    end

    x = copy(x0)
    r = b - (A * x)
    d = copy(r)

    while !RemainingStoppingCriteria(r, b, tol)
        if method == IterativeMethods.JacobiMethod || method == IterativeMethods.GaussSeidelMethod
            x, r = method(A, b, x, P, w, alpha)
        elseif method == IterativeMethods.ConjugateGradient
            x, r, d = method(A, b, x, r, d)
        else
            x, r = method(A, b, x)
        end
        # println("Residual: ", norm(r) / norm(b))
        # r = b - (A * x)
        number_of_iterations += 1
        if number_of_iterations >= max_iter
            println("The method did not converge")
            return x, number_of_iterations
        end
    end

    return x, number_of_iterations
end

"""
    JacobiMethod(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64}, x::Vector{Float64}, P::SparseMatrixCSC, w::Float64, alpha::Float64=1.0)

Solve a linear system using the Jacobi method.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x::Vector{Float64}`: The current solution vector.
- `P::SparseMatrixCSC`: The preconditioner matrix.
- `w::Float64`: The relaxation factor.
- `alpha::Float64`: The parameter for the SOR method.

# Returns
- `Tuple{Vector{Float64},Vector{Float64}}`: The updated solution vector and the residual vector.

"""
function JacobiMethod(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64}, x::Vector{Float64}, P::SparseMatrixCSC, w::Float64, alpha::Float64=1.0)::Tuple{Vector{Float64},Vector{Float64}}
    # Update the solution vector using relaxation:
    # xk = x + alpha * P * (b - A * x) + (1 - w) * x
    r = similar(b)
    mul!(r, A, x)
    r = b - r
    xk = similar(x)
    mul!(xk, P, r)
    xk = x + alpha .* xk + (1 - w) .* x
    return xk, r
end

"""
    GaussSeidelMethod(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64}, x::Vector{Float64}, P::SparseMatrixCSC, w::Float64, alpha::Float64=1.0)

Solve a linear system using the Gauss-Seidel method.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x::Vector{Float64}`: The current solution vector.
- `P::SparseMatrixCSC`: The preconditioner matrix.
- `w::Float64`: The relaxation factor.
- `alpha::Float64`: The parameter for the SOR method.

# Returns
- `Tuple{Vector{Float64},Vector{Float64}}`: The updated solution vector and the residual vector.

"""
function GaussSeidelMethod(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64}, x::Vector{Float64}, P::SparseMatrixCSC, w::Float64, alpha::Float64=1.0)::Tuple{Vector{Float64},Vector{Float64}}
    r = similar(b)
    mul!(r, A, x)
    r = b - r
    y = DirectMethods.ForwardSubstitution(P, r)

    # Update the solution vector using relaxation:
    # xk = x + alpha * y + (1 - w) * x
    xk = x + alpha .* y + (1 - w) .* x
    return xk, r
end

"""
    Gradient(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64}, x::Vector{Float64})

Solve a linear system using the Gradient method.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x::Vector{Float64}`: The current solution vector.

# Returns
- `Tuple{Vector{Float64},Vector{Float64}}`: The updated solution vector and the residual vector.

"""
function Gradient(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64}, x::Vector{Float64})::Tuple{Vector{Float64},Vector{Float64}}
    r = similar(b)
    y = similar(b)

    mul!(r, A, x)
    r = b .- r
    mul!(y, A, r)
    alpha = dot(r, r) / dot(r, y)

    @. x += alpha .* r # Aggiornamento delle iterazioni

    return x, r
end

"""
    ConjugateGradient(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64}, x::Vector{Float64}, r::Vector{Float64}, d::Vector{Float64})

Solve a linear system using the Conjugate Gradient method.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x::Vector{Float64}`: The current solution vector.
- `r::Vector{Float64}`: The residual vector.
- `d::Vector{Float64}`: The direction vector.

# Returns
- `Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}`: The updated solution vector, the residual vector, and the direction vector.

"""
function ConjugateGradient(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64}, x::Vector{Float64}, r::Vector{Float64}, d::Vector{Float64})::Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}
    r = similar(b)
    w = similar(b)
    y = similar(b)

    # r = b - A * x
    mul!(r, A, x)
    r = b - r

    # y = A * d
    mul!(y, A, d)

    # alpha = dot(d, r) / dot(d, y)
    alpha = dot(d, r) / dot(d, y)

    # Aggiorniamo la soluzione x
    @. x += alpha .* d

    mul!(r, A, x)
    rk = b - r

    mul!(w, A, rk)

    beta = dot(d, w) / dot(d, y)

    @. d = rk - beta .* d

    return x, rk, d
end

"""
    jacobi(A::SparseMatrixCSC, b::Vector{Float64}, x0::Vector{Float64}, tol::Float64, maxIter::UInt16)

Solve a linear system using the Jacobi method.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial guess for the solution.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function jacobi(A::SparseMatrixCSC, b::Vector{Float64}, x0::Vector{Float64}, tol::Float64, maxIter::UInt16)::Tuple{Vector{Float64},UInt16}
    return GenericIterativeMethod(A, b, x0, tol, maxIter, IterativeMethods.JacobiMethod, 1.0, 1.0)
end

"""
    gauss_seidel(A::SparseMatrixCSC, b::Vector{Float64}, x0::Vector{Float64}, tol::Float64, maxIter::UInt16)

Solve a linear system using the Gauss-Seidel method.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial guess for the solution.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function gauss_seidel(A::SparseMatrixCSC, b::Vector{Float64}, x0::Vector{Float64}, tol::Float64, maxIter::UInt16)::Tuple{Vector{Float64},UInt16}
    return GenericIterativeMethod(A, b, x0, tol, maxIter, IterativeMethods.GaussSeidelMethod, 1.0, 1.0)
end

"""
    gradient(A::SparseMatrixCSC, b::Vector{Float64}, x0::Vector{Float64}, tol::Float64, maxIter::UInt16)

Solve a linear system using the Gradient method.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial guess for the solution.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function gradient(A::SparseMatrixCSC, b::Vector{Float64}, x0::Vector{Float64}, tol::Float64, maxIter::UInt16)::Tuple{Vector{Float64},UInt16}
    return GenericIterativeMethod(A, b, x0, tol, maxIter, IterativeMethods.Gradient, 1.0, 1.0)
end

"""
    conjugate_gradient(A::SparseMatrixCSC, b::Vector{Float64}, x0::Vector{Float64}, tol::Float64, maxIter::UInt16)

Solve a linear system using the Conjugate Gradient method.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial guess for the solution.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function conjugate_gradient(A::SparseMatrixCSC, b::Vector{Float64}, x0::Vector{Float64}, tol::Float64, maxIter::UInt16)::Tuple{Vector{Float64},UInt16}
    return GenericIterativeMethod(A, b, x0, tol, maxIter, IterativeMethods.ConjugateGradient, 1.0, 1.0)
end


end