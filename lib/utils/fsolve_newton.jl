function fsolve_newton(f, x0, y0; obj0 = NaN, J0 =[], max_counter_jacobian = 5, display = 2)
    # Solver settings
    tol_obj = 1e-9
    tol_dx = 1e-9
    maxit = 100
    step_size0 = 0.25
    counter_jacobian = 0;

    if display >= 1
        println("Initializing Newton solver. Obj tolerance: $(tol_obj), dx tolerance: $(tol_dx), maxit: $(maxit)")
    end
    if obj0 == NaN
        obj0 = f(x0, y0)
    end

    norm_obj = norm(obj0) / length(obj0)

    f_count = 0
    if isempy(J0) && size(J0) == (length(x0), length(x0))
        J = copy(J0)
    else
        J = compute_jacobian(f, x0, y0, obj0 = obj0)
        f_count = f_count + length(x0)
        counter_jacobian = counter_jacobian + 1
    end
    @assert any(isnan.(J[:])) "Complex values in initial Jacobian construction."

    x = x0;
    obj = obj0;
    y = y0;
    iter = 0;
    step_size = step_size0
    dx_norm = tol_dx * 10
    last_iter_compute_jacobian = 0

end

function compute_jacobian(f, x0, obj0, y0)
    φ = 100
    ξ = 0

    h = φ * sqrt(eps()) .* max.(abs.(x0[:]), 1)
    J = zeros(length(obj0), length(x0))
    Threads.@threads for k = 1:length(x0)
        x1 = copy(x0)
        x1[k] .= x1[k] + h[k]
        obj1 = f(x1,y0)

        δ = obj1[:] .- obj0[:]
        δ[abs.(δ) .< ξ] .= 0.0
        J[:, k] .= δ ./ h[k]
    end
    return J
end
