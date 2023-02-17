using SparseArrays

const DenseOrSparseMatrix{T} = Union{Matrix{T},SparseMatrixCSC{T,Int}}

function solveLCP(M::DenseOrSparseMatrix,q::Vector{Float64},x0::Vector{Float64};
                  l::Vector{Float64},
                  u::Vector{Float64},
                  tol= 1e-12,
                  μ = 1e-3,
                  μ_step = 5.0,
                  μmin = 1e-5,
                  max_iter = 10,
                  b_tol = 1e-6)

    n     = size(M,1)
    lu    = [l u]
    x     = copy(x0)
    nx    = copy(x)
    new_x = true
    bad   = BitArray(undef,n)
    good  = BitArray(undef,n)
    ψ,φ,J = FB(x,q,M,l,u)

    for iter = 1:max_iter

        if new_x
            mlu,ilu  = findmin([abs.(x-l) abs.(u-x)],dims=2)
            ilu      = map(i->i[2], ilu)
            ba       = @. max(abs(φ),mlu) < b_tol

            bad     .= ba[:]
            good    .= BitArray(1 .- bad)
            ψ       -= 0.5*φ[bad]'*φ[bad]
            J        = J[good,good] # dimension reduction
            φ        = φ[good]
            new_x    = false
            nx[bad] .= lu[findall(bad) .+ (ilu[bad] .- 1).*n]
        end
        
        H    = J'*J .+ μ .* sparse(I, sum(good),sum(good))
        Jphi = J'*φ        
        d    = -H \ Jphi

        nx[good].= x[good] .+ d
        nψ,nφ,nJ = FB(nx,q,M,l,u)
        r        = (ψ - nψ)  / -(Jphi'*d .+ 0.5*d'*H*d)  # actual reduction / expected reduction

        if r < 0.3           # small reduction, increase mu
            μ = max(μ*μ_step,μmin)
        end

        if r > 0             # some reduction, accept nx
            x   = copy(nx)
            ψ   = copy(nψ)
            φ   = copy(nφ)
            J   = copy(nJ)
            new_x = true
            if r > 0.8       # large reduction, decrease mu
                μ =  μ/μ_step * (μ > μmin)
            end      
        end

        if ψ < tol
            break
        end
    end

    x = min.(max.(x,l),u)

end

function FB(x::Vector{Float64},
            q::Vector{Float64},
            M::DenseOrSparseMatrix,
            l::Vector{Float64},
            u::Vector{Float64})

    n   = length(x)
    Zl  = @. (l > -Inf) & (u==Inf)
    Zu  = @. (l == -Inf) & (u < Inf)
    Zlu = @. (l > -Inf) & (u < Inf)
    Zf  = @. (l == -Inf) & (u == Inf)

    a = copy(x)
    b = M*x .+ q

    a[Zl] .= x[Zl] .- l[Zl]
    # a   .= x .- l
    a[Zu] .= u[Zu] .- x[Zu]
    b[Zu] .*= -1

    if any(Zlu)
        nt     = sum(Zlu)
        at     = u[Zlu] .- x[Zlu]
        bt     = -b[Zlu]
        st     = @. sqrt(at^2 + bt^2)
        a[Zlu] .= x[Zlu] .- l[Zlu]
        b[Zlu] .= st .- at .- bt
    end

    s      = @. sqrt(a^2 + b^2)
    φ      = s .- a .- b
    φ[Zu] .*= -1
    φ[Zf] .= -b[Zf]
    ψ      = 0.5 * φ'*φ

    if any(Zlu)
        M[Zlu,:] .= -sparse(1:nt,findall(Zlu),at./st .- ones(nt),nt,n) - spdiagm(bt./st .- ones(nt))*M[Zlu,:]
    end
    da      = a./s .- ones(n)
    db      = b./s .- ones(n)
    da[Zf] .= 0
    db[Zf] .= -1
    J       = spdiagm(da) .+ spdiagm(db)*M

    return ψ,φ,J
end
