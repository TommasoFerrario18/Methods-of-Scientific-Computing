module IterativeMethods
include("Utils.jl")
include("DirectMethods.jl")

using .Utils
using .DirectMethods
using LinearAlgebra
using SparseArrays

"""
    IncreaseStoppingCriteria(x, x_old, x_0, tol)

Check if the stopping criteria for an iterative method has been reached.

# Arguments
- `x::Vector{Float64}`: The current solution vector.
- `x_old::Vector{Float64}`: The previous solution vector.
- `x_0::Vector{Float64}`: The initial solution vector.
- `tol::Float64`: The tolerance value.

# Returns
- `Bool`: `true` if the stopping criteria has been reached, `false` otherwise.

"""
function IncreaseStoppingCriteria(x::Vector{Float64}, x_old::Vector{Float64},
    x_0::Vector{Float64}, tol::Float64)::Bool
    return norm(x - x_old) / norm(x_0) < tol
end

"""
    DecreaseStoppingCriteria(x, x_old, x_0, tol)

Check if the stopping criteria for an iterative method has been reached.

# Arguments
- `x::Vector{Float64}`: The current solution vector.
- `x_old::Vector{Float64}`: The previous solution vector.
- `x_0::Vector{Float64}`: The initial solution vector.
- `tol::Float64`: The tolerance value.

# Returns
- `Bool`: `true` if the stopping criteria has been reached, `false` otherwise.

"""
function RemainingStoppingCriteria(A::SparseMatrixCSC{Float64,UInt32},
    x::Vector{Float64}, b::Vector{Float64}, tol::Float64)::Bool
    return norm(b - (A * x)) / norm(b) < tol
end

"""
    RelaxedJacobi(A, b, x0, tol, maxIter, w, stoppingCriterion, alpha)

Solve a linear system using the Jacobi method with relaxation.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial solution vector.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.
- `w::Float64`: The relaxation factor.
- `stoppingCriterion::Function`: The stopping criterion function.
- `alpha::Float64`: The Richardson parameter.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function RelaxedJacobi(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    x0::Vector{Float64}, tol::Float64, maxIter::UInt16, w::Float64,
    stoppingCriterion::Function=IncreaseStoppingCriteria, alpha::Float64=1.0)::Tuple{Vector{Float64},UInt16}
    n = size(A)[1]

    if n != size(A)[2] || n != size(b)[1] || n != size(x0)[1]
        throw(ArgumentError("The dimensions of the input matrices are not compatible"))
    end

    if w <= 0 || w > 1
        throw(ArgumentError("The relaxation factor must be in the interval (0, 1]"))
    end

    k = 0
    x = copy(x0)
    xk = zeros(n)

    P = sparse(I, n, n)

    for i = 1:n
        P[i, i] = w / A[i, i]
    end

    while k < maxIter
        xk = x + alpha * P * (b - A * x) + (1 - w) * x

        if stoppingCriterion(xk, x, x0, tol)
            return xk, k
        end

        x = copy(xk)
        k += 1
    end
    println("The method did not converge")
    return xk, k
end

"""
    Jacobi(A, b, x0, tol, maxIter, stoppingCriterion)

Solve a linear system using the Jacobi method.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial solution vector.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.
- `stoppingCriterion::Function`: The stopping criterion function.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function Jacobi(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    x0::Vector{Float64}, tol::Float64, maxIter::UInt16,
    stoppingCriterion::Function=IncreaseStoppingCriteria)::Tuple{Vector{Float64},UInt16}
    return RelaxedJacobi(A, b, x0, tol, maxIter, 1.0, stoppingCriterion)
end

"""
    Jacobi(A, b, tol, maxIter, stoppingCriterion)

Solve a linear system using the Jacobi method.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.
- `stoppingCriterion::Function`: The stopping criterion function.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function Jacobi(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    tol::Float64, maxIter::UInt16, stoppingCriterion::Function=IncreaseStoppingCriteria)::Tuple{Vector{Float64},UInt16}
    return RelaxedJacobi(A, b, zeros(size(b)), tol, maxIter, 1.0, stoppingCriterion)
end

"""
    Jacobi(A, b, tol, maxIter, w, stoppingCriterion)

Solve a linear system using the Jacobi method with relaxation.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.
- `w::Float64`: The relaxation factor.
- `stoppingCriterion::Function`: The stopping criterion function.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function RelaxedJacobi(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    tol::Float64, maxIter::UInt16, w::Float64, stoppingCriterion::Function=
    IncreaseStoppingCriteria)::Tuple{Vector{Float64},UInt16}
    return RelaxedJacobi(A, b, zeros(size(b)), tol, maxIter, w, stoppingCriterion)
end

"""
    RelaxedGaussSeidel(A, b, x0, tol, maxIter, w, stoppingCriterion, alpha)

Solve a linear system using the Gauss-Seidel method with relaxation.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial solution vector.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.
- `w::Float64`: The relaxation factor.
- `stoppingCriterion::Function`: The stopping criterion function.

# Returns 
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function RelaxedGaussSeidel(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    x0::Vector{Float64}, tol::Float64, maxIter::UInt16, w::Float64,
    stoppingCriterion::Function=stoppingCriterion, alpha::Float64=1.0)::Tuple{Vector{Float64},UInt16}
    n = size(A)[1]

    if n != size(A)[2] || n != size(b)[1] || n != size(x0)[1]
        throw(ArgumentError("The dimensions of the input matrices are not compatible"))
    end

    if w <= 0 || w >= 2
        throw(ArgumentError("The relaxation factor must be in the interval (0, 2)"))
    end

    k = 0
    x = copy(x0)
    xk = zeros(n)

    P = tril(A)

    for i = 1:n
        P[i, i] = w / P[i, i]
    end

    while k < maxIter
        r = b - A * x
        y = DirectMethods.forwardSubstitution(P, r)

        xk = x + alpha * y + (1 - w) * x

        if stoppingCriterion(x, xk, x0, tol)
            return xk, k
        end

        x = copy(xk)
        k += 1
    end
    println("The method did not converge")
    return xk, k
end

"""
    GaussSeidel(A, b, x0, tol, maxIter, stoppingCriterion)

Solve a linear system using the Gauss-Seidel method.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial solution vector.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.
- `stoppingCriterion::Function`: The stopping criterion function.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function GaussSeidel(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64}, x0::Vector{Float64},
    tol::Float64, maxIter::UInt16, stoppingCriterion::Function=IncreaseStoppingCriteria)::Tuple{Vector{Float64},UInt16}
    return RelaxedGaussSeidel(A, b, x0, tol, maxIter, 1.0, stoppingCriterion)
end

"""
    GaussSeidel(A, b, tol, maxIter, stoppingCriterion)

Solve a linear system using the Gauss-Seidel method.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.
- `stoppingCriterion::Function`: The stopping criterion function.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function GaussSeidel(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64}, tol::Float64,
    maxIter::UInt16, stoppingCriterion::Function=IncreaseStoppingCriteria)::Tuple{Vector{Float64},UInt16}
    return RelaxedGaussSeidel(A, b, zeros(size(b)), tol, maxIter, 1.0, stoppingCriterion)
end

"""
    GaussSeidel(A, b, tol, maxIter, w, stoppingCriterion)

Solve a linear system using the Gauss-Seidel method with relaxation.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.
- `w::Float64`: The relaxation factor.
- `stoppingCriterion::Function`: The stopping criterion function.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function RelaxedGaussSeidel(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64}, tol::Float64,
    maxIter::UInt16, w::Float64, stoppingCriterion::Function=IncreaseStoppingCriteria)::Tuple{Vector{Float64},UInt16}
    return RelaxedGaussSeidel(A, b, zeros(size(b)), tol, maxIter, w, stoppingCriterion)
end

"""
    RichardsonJacobi(A, b, x0, tol, maxIter, w, stoppingCriterion, alpha)

Solve a linear system using the Richardson method with relaxation.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial solution vector.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.
- `w::Float64`: The relaxation factor.
- `stoppingCriterion::Function`: The stopping criterion function.
- `alpha::Float64`: The Richardson parameter.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function RichardsonJacobi(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    x0::Vector{Float64}, tol::Float64, maxIter::UInt16, w::Float64,
    stoppingCriterion::Function=IncreaseStoppingCriteria, alpha::Float64=1.0)::Tuple{Vector{Float64},UInt16}
    if alpha <= 0
        throw(ArgumentError("The parameter alpha must be positive"))
    end

    # TODO: Inserire la possibilità di calcolare alpha ottimale, su scelta 
    # TODO: dell'utente perchè è un calcolo costoso

    return RelaxedJacobi(A, b, x0, tol, maxIter, w, stoppingCriterion, alpha)
end

"""
    RichardsonGaussSeidel(A, b, x0, tol, maxIter, w, stoppingCriterion, alpha)

Solve a linear system using the Richardson method with relaxation.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial solution vector.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.
- `w::Float64`: The relaxation factor.
- `stoppingCriterion::Function`: The stopping criterion function.
- `alpha::Float64`: The Richardson parameter.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function RichardsonGaussSeidel(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    x0::Vector{Float64}, tol::Float64, maxIter::UInt16, w::Float64,
    stoppingCriterion::Function=IncreaseStoppingCriteria, alpha::Float64=1.0)::Tuple{Vector{Float64},UInt16}
    if alpha <= 0
        throw(ArgumentError("The parameter alpha must be positive"))
    end

    # TODO: Inserire la possibilità di calcolare alpha ottimale, su scelta 
    # TODO: dell'utente perchè è un calcolo costoso

    return RelaxedGaussSeidel(A, b, x0, tol, maxIter, w, stoppingCriterion,
        alpha)
end

"""
    Gradient(A, b, x0, tol, maxIter, stoppingCriterion)

Solve a linear system using the Gradient method.

# Arguments
- `A::SparseMatrixCSC{Float64,Int64}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial solution vector.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.
- `stoppingCriterion::Function`: The stopping criterion function.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function Gradient(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    x0::Vector{Float64}, tol::Float64, maxIter::UInt16, stoppingCriterion::Function=IncreaseStoppingCriteria)::Tuple{Vector{Float64},UInt16}
    n = size(A)[1]

    if n != size(A)[2] || n != size(b)[1] || n != size(x0)[1]
        throw(ArgumentError("The dimensions of the input matrices are not compatible"))
    end

    k = 0
    x = copy(x0)

    while k < maxIter
        r = b - A * x
        y = A * r
        alpha = dot(r, r) / dot(r, y)

        xk = x + alpha * r
        if stoppingCriterion(x, xk, x0, tol)
            println("Converged in $k iterations")
            return xk, k
        end
        x = copy(xk)
        k += 1
    end
    println("The method did not converge")
    return x, k
end

function ConjugateGradient(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    x0::Vector{Float64}, tol::Float64, maxIter::UInt16, stoppingCriterion::Function=IncreaseStoppingCriteria)::Tuple{Vector{Float64},UInt16}

    n = size(A)[1]

    if n != size(A)[2] || n != size(b)[1] || n != size(x0)[1]
        throw(ArgumentError("The dimensions of the input matrices are not compatible"))
    end

    k = 0
    x = copy(x0)
    r = b - A * x
    d = copy(r)

    while k < maxIter
        r = b - A * x
        y = A * d
        z = A * r
        alpha = dot(d, r) / dot(d, y)
        xk = x + alpha * d

        rk = b - A * xk
        w = A * rk

        beta = dot(d, w) / dot(d, y)
        d = rk - beta * d

        if stoppingCriterion(x, xk, x0, tol)
            println("Converged in $k iterations")
            return xk, k
        end

        x = copy(xk)
        k += 1
    end
    println("The method did not converge")
    return x, k
end
end