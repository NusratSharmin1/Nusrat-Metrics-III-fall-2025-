using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions
cd(@__DIR__)

# Ps2_tests_Nusrat.jl

# reproducibility
Random.seed!(42)

# ---------------------------
# Helpers (used across tests)
# ---------------------------
const URL_NLSW = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"

ols_ssr(beta, X, y) = (y .- X*beta)' * (y .- X*beta)

function logit_nll(alpha, X, y::AbstractVector{<:Integer})
    η = X*alpha
    p = 1 ./(1 .+ exp.(-η))
    ϵ = 1e-12
    p = clamp.(p, ϵ, 1-ϵ)
    return -sum(y .* log.(p) .+ (1 .- y) .* log.(1 .- p))
end

function mlogit_nll(α::AbstractVector, X::AbstractMatrix, y::AbstractVector{<:Integer})
    N, K = size(X)
    J = length(unique(y))
    A = reshape(α, K, J-1)
    U = X*A
    Z = hcat(U, zeros(N))
    m = maximum(Z, dims=2)
    lse = vec(m) .+ log.(sum(exp.(Z .- m), dims=2))
    chosen_u = Z[CartesianIndex.(collect(1:N), y)]
    return -sum(chosen_u .- lse)
end

function mlogit_probs(α::AbstractVector, X::AbstractMatrix)
    N, K = size(X)
    Jm1 = length(α) ÷ K
    A = reshape(α, K, Jm1)
    Z = hcat(X*A, zeros(N))
    m = maximum(Z, dims=2)
    numer = exp.(Z .- m)
    denom = sum(numer, dims=2)
    return numer ./ denom
end

# ============================
#            TESTS
# ============================
@testset "Problem Set 2 — Nusrat Tests" begin

    # Q1
    @testset "Q1: Optimization" begin
        f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2
        minusf(x) = -f(x)
        res = optimize(minusf, [-7.0], BFGS())
        xhat = Optim.minimizer(res)[1]
        @test isapprox(xhat, -7.3782434055; atol=1e-4)
        @test f([xhat]) ≥ f([xhat - 1e-3]) && f([xhat]) ≥ f([xhat + 1e-3])
        @test isapprox(Optim.minimum(res), minusf([xhat]); atol=1e-10)
    end

    # Q2
    @testset "Q2: OLS" begin
        df = CSV.read(HTTP.get(URL_NLSW).body, DataFrame)
        X = [ones(size(df,1),1) df.age df.race .== 1 df.collgrad .== 1]
        y = df.married .== 1
        β̂_opt = optimize(b -> ols_ssr(b, X, y), rand(size(X,2)), LBFGS()).minimizer
        β̂_cf  = (X'X) \ (X'y)
        @test isapprox(β̂_opt, β̂_cf; atol=1e-6)
    end

    # Q3
    @testset "Q3: Logit" begin
        df = CSV.read(HTTP.get(URL_NLSW).body, DataFrame)
        df.white = df.race .== 1
        X = [ones(size(df,1),1) df.age df.white df.collgrad]
        y = df.married .== 1
        m_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
        β_glm = coef(m_glm)
        res = optimize(a -> logit_nll(a, X, y), zeros(size(X,2)), BFGS(); autodiff = :forward)
        α̂ = Optim.minimizer(res)
        @test isapprox(α̂, β_glm; atol=1e-3, rtol=1e-3)
    end

    # Q4
    @testset "Q4: GLM predictions" begin
        df = CSV.read(HTTP.get(URL_NLSW).body, DataFrame)
        df.white = df.race .== 1
        m = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
        p = predict(m)
        @test all(0 .≤ p .≤ 1)
    end

    # Q5
    @testset "Q5: Multinomial Logit" begin
        df = CSV.read(HTTP.get(URL_NLSW).body, DataFrame)
        df = dropmissing(df, :occupation)
        for j in (8,9,10,11,12,13)
            df[df.occupation .== j, :occupation] .= 7
        end
        X = [ones(nrow(df),1) df.age df.race .== 1 df.collgrad .== 1]
        y = convert(Vector{Int}, df.occupation)
        N, K = size(X)
        J = length(unique(y))
        α0 = zeros(K*(J-1))
        f0 = mlogit_nll(α0, X, y)
        res = optimize(a -> mlogit_nll(a, X, y), α0, BFGS(); g_tol=1e-5, autodiff=:forward)
        α̂ = Optim.minimizer(res)
        f̂ = Optim.minimum(res)
        @test f̂ < f0
        P = mlogit_probs(α̂, X)
        rowsums = sum(P, dims=2)
        @test all(abs.(rowsums .- 1) .< 1e-8)
    end
end



















