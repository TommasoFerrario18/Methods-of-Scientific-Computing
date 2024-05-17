using LinearAlgebra

function new_base(n)
    base = zeros(n, n)
    for k in 1:n
        for i in 1:n
            base[k, i] = cos(k * pi * ((2 * i + 1) / 2 * n))
        end
    end
    return base
end


