function newton_nonlin(f, fprime, x0, a, crit)

    x = copy(x0); dx = size(x0); it = 1;
    while maximum(dx) > crit || maximum(abs.(f(x, a))) > crit
        xnew = x .- (f(x, a) ./ fprime(x))
        dx = abs.(x .- xnew)
        x .= xnew
        it += 1
    end
    return x
end
