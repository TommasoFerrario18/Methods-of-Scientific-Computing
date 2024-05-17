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
    RelaxedJacobi(A, b, x0, tol, maxIter, w, alpha)

Solve a linear system using the Jacobi method with relaxation.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial solution vector.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.
- `w::Float64`: The relaxation factor.
- `alpha::Float64`: The Richardson parameter.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function RelaxedJacobi(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    x0::Vector{Float64}, tol::Float64, maxIter::UInt16, w::Float64, alpha::Float64=1.0)::Tuple{Vector{Float64},UInt16}
    n = size(A)[1]

    if n != size(A)[2] || n != size(b)[1] || n != size(x0)[1]
        throw(ArgumentError("The dimensions of the input matrices are not compatible"))
    end

    if w <= 0 || w > 1
        throw(ArgumentError("The relaxation factor must be in the interval (0, 1]"))
    end

    # Number of iterations
    k = 0

    # Solution vector
    x = copy(x0)  # Current solution vector
    xk = zeros(n) # Next solution vector

    # Residual vector
    r = zeros(n)

    # Diagonal matrix of A and inverted
    P = spdiagm(0 => w ./ diag(A))

    while k < maxIter
        # Update the solution vector using relaxation:
        # xk = x + alpha * P * (b - A * x) + (1 - w) * x
        mul!(r, A, x)
        r = b - r
        mul!(xk, P, r)
        xk = x + alpha .* xk + (1 - w) .* x

        if RemainingStoppingCriteria(r, b, tol)
            return xk, k
        end

        copyto!(x, xk)
        k += 1
    end
    println("The method did not converge")
    return xk, k
end

"""
    Jacobi(A, b, x0, tol, maxIter)

Solve a linear system using the Jacobi method.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial solution vector.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function Jacobi(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    x0::Vector{Float64}, tol::Float64, maxIter::UInt16)::Tuple{Vector{Float64},UInt16}
    return RelaxedJacobi(A, b, x0, tol, maxIter, 1.0)
end


"""
    RelaxedGaussSeidel(A, b, x0, tol, maxIter, w, alpha)

Solve a linear system using the Gauss-Seidel method with relaxation.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial solution vector.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.
- `w::Float64`: The relaxation factor.

# Returns 
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function RelaxedGaussSeidel(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    x0::Vector{Float64}, tol::Float64, maxIter::UInt16, w::Float64, alpha::Float64=1.0)::Tuple{Vector{Float64},UInt16}
    n = size(A)[1]

    if n != size(A)[2] || n != size(b)[1] || n != size(x0)[1]
        throw(ArgumentError("The dimensions of the input matrices are not compatible"))
    end

    if w <= 0 || w >= 2
        throw(ArgumentError("The relaxation factor must be in the interval (0, 2)"))
    end

    # Number of iterations
    k = 0

    # Solution vector
    x = copy(x0)  # Current solution vector
    xk = copy(x0) # Next solution vector

    # Compute the lower triangular matrix of A
    P = sparse(tril(A))

    for i = 1:n
        P[i, i] = P[i, i] / w
    end

    r = similar(b)
    y = similar(b)

    while k < maxIter
        mul!(r, A, x)
        r = b - r
        y = DirectMethods.ForwardSubstitution(P, r)

        # Update the solution vector using relaxation:
        # xk = x + alpha * y + (1 - w) * x
        xk = x + alpha .* y + (1 - w) .* x

        if RemainingStoppingCriteria(r, b, tol)
            return xk, k
        end

        copyto!(x, xk)
        k += 1
    end
    println("The method did not converge")
    return xk, k
end

"""
    GaussSeidel(A, b, x0, tol, maxIter)

Solve a linear system using the Gauss-Seidel method.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial solution vector.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function GaussSeidel(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64}, x0::Vector{Float64},
    tol::Float64, maxIter::UInt16)::Tuple{Vector{Float64},UInt16}
    return RelaxedGaussSeidel(A, b, x0, tol, maxIter, 1.0)
end

"""
    RichardsonJacobi(A, b, x0, tol, maxIter, w, alpha)

Solve a linear system using the Richardson method with relaxation.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial solution vector.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.
- `w::Float64`: The relaxation factor.
- `alpha::Float64`: The Richardson parameter.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function RichardsonJacobi(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    x0::Vector{Float64}, tol::Float64, maxIter::UInt16, w::Float64, alpha::Float64=1.0,
    choose_alpha::Bool=false)::Tuple{Vector{Float64},UInt16}
    if choose_alpha
        P = sparse(1.0 * I, size(A)[1], size(A)[2])

        for i = 1:size(A)[1]
            P[i, i] = w / A[i, i]
        end

        alpha = Utils.optimal_alpha_richardson(P)
    end

    if alpha <= 0
        throw(ArgumentError("The parameter alpha must be positive"))
    end

    return RelaxedJacobi(A, b, x0, tol, maxIter, w, alpha)
end

"""
    RichardsonGaussSeidel(A, b, x0, tol, maxIter, w, alpha)

Solve a linear system using the Richardson method with relaxation.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial solution vector.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.
- `w::Float64`: The relaxation factor.
- `alpha::Float64`: The Richardson parameter.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function RichardsonGaussSeidel(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    x0::Vector{Float64}, tol::Float64, maxIter::UInt16, w::Float64, alpha::Float64=1.0,
    choose_alpha::Bool=false)::Tuple{Vector{Float64},UInt16}

    if choose_alpha
        P = tril(A)

        for i = 1:size(A)[1]
            P[i, i] = P[i, i] / w
        end

        alpha = Utils.optimal_alpha_richardson(inv(P))
    end

    if alpha <= 0
        throw(ArgumentError("The parameter alpha must be positive"))
    end

    return RelaxedGaussSeidel(A, b, x0, tol, maxIter, w, alpha)
end

"""
    Gradient(A, b, x0, tol, maxIter)

Solve a linear system using the Gradient method.

# Arguments
- `A::SparseMatrixCSC{Float64,Int64}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial solution vector.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function Gradient(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    x0::Vector{Float64}, tol::Float64, maxIter::UInt16)::Tuple{Vector{Float64},UInt16}
    n = size(A)[1]

    if n != size(A)[2] || n != size(b)[1] || n != size(x0)[1]
        throw(ArgumentError("The dimensions of the input matrices are not compatible"))
    end

    k = 0
    x = copy(x0)

    r = similar(b)
    y = similar(b)

    while k < maxIter
        mul!(r, A, x)
        r = b .- r
        mul!(y, A, r)
        alpha = dot(r, r) / dot(r, y)

        @. x += alpha * r # Aggiornamento delle iterazioni

        if RemainingStoppingCriteria(A, x, b, tol)
            return x, k
        end

        k += 1
    end
    println("The method did not converge")
    return x, k
end

"""
    ConjugateGradient(A, b, x0, tol, maxIter)

Solve a linear system using the Conjugate Gradient method.

# Arguments
- `A::SparseMatrixCSC{Float64,UInt32}`: The matrix of the linear system.
- `b::Vector{Float64}`: The right-hand side of the linear system.
- `x0::Vector{Float64}`: The initial solution vector.
- `tol::Float64`: The tolerance value.
- `maxIter::UInt16`: The maximum number of iterations.

# Returns
- `Tuple{Vector{Float64},UInt16}`: The solution vector and the number of iterations.

"""
function ConjugateGradient(A::SparseMatrixCSC{Float64,UInt32}, b::Vector{Float64},
    x0::Vector{Float64}, tol::Float64, maxIter::UInt16)::Tuple{Vector{Float64},UInt16}

    n = size(A)[1]

    if n != size(A)[2] || n != size(b)[1] || n != size(x0)[1]
        throw(ArgumentError("The dimensions of the input matrices are not compatible"))
    end

    k = 0
    x = copy(x0)
    r = b - A * x # Residuo del passo k
    d = copy(r) # Direzione di discesa

    y = similar(b)
    w = similar(b)

    while k < maxIter
        mul!(r, A, x)
        r = b - r
        mul!(y, A, d)
        alpha = dot(d, r) / dot(d, y)

        @. x += alpha .* d

        mul!(r, A, x)
        rk = b - r

        mul!(w, A, rk)

        beta = dot(d, w) / dot(d, y)

        @. d = rk - beta .* d

        if RemainingStoppingCriteria(A, x, b, tol)
            return x, k
        end

        k += 1
    end
    return x, k
end
end