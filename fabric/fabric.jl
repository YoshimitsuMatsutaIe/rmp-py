using ForwardDiff: gradient, jacobian, derivative
using Zygote
using LinearAlgebra
using Plots
using DifferentialEquations

s(v) = v < 0 ? 1.0 : 0.0


function jvp(f, x, u)
    return ForwardDiff.derivative(t->f(x + t*u), 0.0)
end

attractor_task_map(q, qd) = q - qd
repeller_task_map(q, qo, r) = norm(q - qo) / r - 1


function attractor_fabric(x, ẋ)
    k = 150.0
    αᵩ = 10
    β = 100.5
    K = 10
    m₊ = 2.0
    m₋ = 0.2
    αₘ = 0.75

    # ψ₁(x) = k * (norm(x) + (1/αᵩ)*log(1+exp(-2αᵩ*norm(x))))
    # δx = ForwardDiff.gradient(ψ₁, x)
    # #ẍ = -K*δx - β*ẋ
    # ẍ = -δx
    G(x) = (m₊ - m₋) * exp(-(αₘ*norm(x))^2) * I(2) + m₋ * I(2)

    function ℒ(x, ẋ)
        1/2 * ẋ' * G(x) * ẋ
    end

    ξ = -jacobian(gradient(ℒ, ẋ), x) - gradient(ℒ, x)
    
    # ξ = -jacobian(gradient(ℒ, ẋ), x)
    # ξ = ξ - gradient(ℒ, x)

    # _M = gradient(ℒ, ẋ)
    # M = jacobian(gradient(ℒ, ẋ), ẋ) 
    # ẍ = ξ
    return (M, ẍ)
end


function myODE!(dQ, Q, p, t)
    q = Q[1:2]
    q̇ = Q[3:4]

    qd = [1, 1]

    x = attractor_task_map(q, qd)
    ẋ = q̇
    M, f = attractor_fabric(x, ẋ)

    q̈ = pinv(M) * f

    dQ[1:2] = q̇
    dQ[3:4] = q̈

    nothing
end

x_init = [0, 0, 0, 0]
t_span = (0, 50)
prob = ODEProblem(myODE!, x_init, t_span)
sol = solve(prob)


plot(sol)
savefig("sb2_1.png")

plot(sol, vars=(1,2))
savefig("sb2_2.png")



# function repeller_fabric(x, ẋ)
#     kᵦ = 75
#     αᵦ = 50.0
#     s = [v < 0 ? 1.0 : 0.0 for v in ẋ]
#     M = diagm((s.*kᵦ) ./ (x.^2))
#     ψ(θ) = αᵦ ./ (2*θ.^8)
#     x = convert(Vector{Float64}, x)
#     δx = ForwardDiff.jacobian(ψ, x)
#     ẍ = vec((-s .* ẋ.^2)' * δx)
#     return (M, ẍ)
# end

# function fabric_eval(x, ẋ, name::Symbol, env::PointMass)
#     M = nothing; ẍ = nothing
#     fabricname = Symbol(name, :_fabric)
#     ϕ = eval(fabricname)
#     M, ẍ = ϕ(x, ẋ, env)
#     return (M, ẍ)
# end

# function energize(ẍ, M, env::PointMass; ϵ=1e-1)
#     ẋ = env.ẋ + ẍ*env.Δt
#     ẋ = ẋ/(norm(ẋ))
#     ẍₑ = (I(size(M)[1]) - ϵ*ẋ*ẋ')*ẍ
#     return ẍₑ
# end

# function pointmass_fabric_solve(θ, θ̇ , env::PointMass)
#     xₛ = []
#     ẋₛ = []
#     cₛ = []
#     Mₛ = []
#     ẍₛ = []
#     Jₛ = []
#     for t in env.task_maps
#         ψ = eval(Symbol(t, :_task_map))
#         x = ψ(θ, env)
#         ẋ = jvp(σ->ψ(σ, env), θ, θ̇ )
#         c = jvp(σ -> jvp(σ->ψ(σ, env), σ, θ̇  ), θ, θ̇ )
#         J = ForwardDiff.jacobian(σ->ψ(σ, env), θ)
#         M, ẍ = fabric_eval(x, ẋ, t, env)
#         push!(xₛ, x); push!(ẋₛ, ẋ); push!(cₛ, c)
#         push!(Mₛ, M); push!(ẍₛ, ẍ); push!(Jₛ, J)
#     end
#     Mᵣ = sum([J' * M * J for (J, M) in zip(Jₛ, Mₛ)])
#     fᵣ = sum([J' * M * (ẍ - c) for (J, M, ẍ, c) in zip(Jₛ, Mₛ, ẍₛ, cₛ)])
#     Mᵣ = convert(Matrix{Float64}, Mᵣ)
#     ẍ = pinv(Mᵣ) * fᵣ
#     ẍ = energize(ẍ, Mᵣ, env)
#     return ẍ
# end