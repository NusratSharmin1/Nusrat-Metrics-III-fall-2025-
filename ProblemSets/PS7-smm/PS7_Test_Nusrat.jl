using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

cd(@__DIR__)

 include("PS7_Source_Nusrat.jl")


@testset "PS7 GMM and SMM Tests" begin
    
    #--------------------------------------------------------------------------
    # Test 1: OLS GMM Function
    #--------------------------------------------------------------------------
    @testset "OLS GMM" begin
        # Create simple test data
        Random.seed!(123)
        N = 100
        X = [ones(N) randn(N)]
        β_true = [2.0, 0.5]
        y = X * β_true + 0.1 * randn(N)
        
        # Test that GMM objective is minimized at OLS estimate
        β_ols = X \ y
        obj_at_ols = ols_gmm(β_ols, X, y)
        obj_at_wrong = ols_gmm([0.0, 0.0], X, y)
        
        @test obj_at_ols < obj_at_wrong
        @test obj_at_ols ≈ 0.0 atol=2.0  # Should be close to zero
        
        # Test with multiple covariates
        X_multi = [ones(N) randn(N, 3)]
        β_multi = randn(4)
        y_multi = X_multi * β_multi + randn(N)
        β_est = X_multi \ y_multi
        @test ols_gmm(β_est, X_multi, y_multi) < ols_gmm(zeros(4), X_multi, y_multi)
    end
    
    #--------------------------------------------------------------------------
    # Test 2: Multinomial Logit Functions
    #--------------------------------------------------------------------------
    @testset "Multinomial Logit MLE" begin
        # Create simple test data
        Random.seed!(456)
        N = 50
        J = 3
        K = 2
        X = [ones(N) randn(N)]
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        # Test that function runs without error
        @test isa(mlogit_mle(α, X, y), Number)
        @test mlogit_mle(α, X, y) > 0  # Log-likelihood should be positive
        
        # Test with different parameter values
        α_zeros = zeros(K * (J-1))
        @test mlogit_mle(α_zeros, X, y) > 0
        @test !isnan(mlogit_mle(α, X, y))
        @test !isinf(mlogit_mle(α, X, y))
    end
    
    @testset "Multinomial Logit GMM" begin
        Random.seed!(789)
        N = 50
        J = 3
        K = 2
        X = [ones(N) randn(N)]
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        # Test that function runs without error
        @test isa(mlogit_gmm(α, X, y), Number)
        @test mlogit_gmm(α, X, y) ≥ 0  # Objective should be non-negative
        
        # Test that moments are well-defined
        @test !isnan(mlogit_gmm(α, X, y))
        @test !isinf(mlogit_gmm(α, X, y))
    end
    
    @testset "Multinomial Logit GMM Overidentified" begin
        Random.seed!(101)
        N = 50
        J = 3
        K = 2
        X = [ones(N) randn(N)]
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        # Test that function runs without error
        @test isa(mlogit_gmm_overid(α, X, y), Number)
        @test mlogit_gmm_overid(α, X, y) ≥ 0
        
        # Test that overidentified GMM gives different value than just-identified
        gmm_just = mlogit_gmm(α, X, y)
        gmm_over = mlogit_gmm_overid(α, X, y)
        @test gmm_just != gmm_over  # Should be different in general
    end
    
    #--------------------------------------------------------------------------
    # Test 3: Data Simulation Functions
    #--------------------------------------------------------------------------
    @testset "Simulate Logit with Inverse CDF" begin
        Random.seed!(111)
        N = 1000
        J = 4
        
        Y, X = sim_logit(N, J)
        
        # Test dimensions
        @test length(Y) == N
        @test size(X, 1) == N
        @test size(X, 2) == 4  # intercept + 3 covariates
        
        # Test that choices are in valid range
        @test all(1 .<= Y .<= J)
        
        # Test that all choices are represented (with high N)
        @test length(unique(Y)) == J
        
        # Test that X has reasonable properties
        @test all(X[:, 1] .== 1)  # First column is intercept
        @test std(X[:, 2]) > 0  # Other columns have variation
    end
    
    @testset "Simulate Logit with Gumbel" begin
        Random.seed!(222)
        N = 1000
        J = 4
        
        Y, X = sim_logit_with_gumbel(N, J)
        
        # Test dimensions
        @test length(Y) == N
        @test size(X, 1) == N
        
        # Test that choices are in valid range
        @test all(1 .<= Y .<= J)
        
        # Test that all choices are represented
        @test length(unique(Y)) == J
        
        # Test data types
        @test eltype(Y) <: Integer
        @test eltype(X) <: AbstractFloat
    end
    
    #--------------------------------------------------------------------------
    # Test 4: Consistency Between Methods
    #--------------------------------------------------------------------------
    @testset "Simulation Methods Produce Similar Results" begin
        Random.seed!(333)
        N = 10000
        J = 4
        
        Y1, X1 = sim_logit(N, J)
        Y2, X2 = sim_logit_with_gumbel(N, J)
        
        # Both methods should produce similar choice frequencies
        freq1 = [mean(Y1 .== j) for j in 1:J]
        freq2 = [mean(Y2 .== j) for j in 1:J]
        
        # Allow 5% tolerance in frequencies
        @test all(abs.(freq1 .- freq2) .< 0.05)
    end
    
    #--------------------------------------------------------------------------
    # Test 5: Optimization Convergence
    #--------------------------------------------------------------------------
    @testset "OLS GMM Converges to True Value" begin
        Random.seed!(444)
        N = 500
        X = [ones(N) randn(N) randn(N)]
        β_true = [1.0, -0.5, 0.3]
        y = X * β_true + 0.5 * randn(N)
        
        result = optimize(b -> ols_gmm(b, X, y), 
                         zeros(3), 
                         LBFGS(), 
                         Optim.Options(g_tol=1e-6))
        
        @test Optim.converged(result)
        @test norm(result.minimizer - β_true) < 0.2  # Should be close to true value
    end
    
    #--------------------------------------------------------------------------
    # Test 6: Edge Cases
    #--------------------------------------------------------------------------
    @testset "Edge Cases" begin
        # Test with minimal data
        X_min = ones(10, 1)
        y_min = randn(10)
        
        @test isa(ols_gmm([1.0], X_min, y_min), Number)
        
        # Test with different J values
        for J in [2, 3, 5]
            Y, X = sim_logit(100, J)
            @test length(unique(Y)) <= J
            @test maximum(Y) <= J
            @test minimum(Y) >= 1
        end
    end
    
    #--------------------------------------------------------------------------
    # Test 7: Parameter Recovery from Simulated Data
    #--------------------------------------------------------------------------
    @testset "Parameter Recovery" begin
        Random.seed!(555)
        N = 5000
        J = 3
        
        # Simulate data
        Y, X = sim_logit_with_gumbel(N, J)
        
        # Estimate using MLE
        K = size(X, 2)
        α_init = zeros(K * (J-1))
        
        result = optimize(α -> mlogit_mle(α, X, Y),
                         α_init,
                         LBFGS(),
                         Optim.Options(iterations=1000))
        
        @test Optim.converged(result)
        
        # Coefficients should be reasonable magnitude
        @test all(abs.(result.minimizer) .< 5.0)
    end
    
    #--------------------------------------------------------------------------
    # Test 8: Numerical Stability
    #--------------------------------------------------------------------------
    @testset "Numerical Stability" begin
        Random.seed!(666)
        N = 100
        J = 3
        K = 2
        X = [ones(N) randn(N)]
        y = rand(1:J, N)
        
        # Test with large coefficients
        α_large = 10.0 * randn(K * (J-1))
        @test !isnan(mlogit_mle(α_large, X, y))
        @test !isinf(mlogit_mle(α_large, X, y))
        
        # Test with small coefficients
        α_small = 0.01 * randn(K * (J-1))
        @test !isnan(mlogit_mle(α_small, X, y))
        @test !isinf(mlogit_mle(α_small, X, y))
    end
    
    #--------------------------------------------------------------------------
    # Test 9: Data Loading Functions
    #--------------------------------------------------------------------------
    @testset "Data Loading" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        
        df, X, y = load_data(url)
        
        # Test that data loaded correctly
        @test size(df, 1) > 0
        @test size(X, 1) == size(df, 1)
        @test length(y) == size(df, 1)
        @test size(X, 2) == 4  # intercept + age + race + collgrad
        
        # Test that X has intercept
        @test all(X[:, 1] .== 1)
    end
    
    @testset "Occupation Data Preparation" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df = CSV.read(HTTP.get(url).body, DataFrame)
        
        df_clean, X, y = prepare_occupation_data(df)
        
        # Test that occupations were collapsed
        @test maximum(skipmissing(df_clean.occupation)) <= 7
        @test minimum(skipmissing(df_clean.occupation)) >= 1
        
        # Test that white variable was created
        @test :white in names(df_clean)
        @test eltype(df_clean.white) == Bool
    end
    
    #--------------------------------------------------------------------------
    # Test 10: Choice Probability Properties
    #--------------------------------------------------------------------------
    @testset "Choice Probability Properties" begin
        Random.seed!(777)
        N = 10000
        J = 4
        
        # Generate choices
        Y, _ = sim_logit_with_gumbel(N, J)
        
        # Probabilities should sum to 1
        freqs = [mean(Y .== j) for j in 1:J]
        @test sum(freqs) ≈ 1.0 atol=0.01
        
        # All probabilities should be positive
        @test all(freqs .> 0)
    end
    
    #--------------------------------------------------------------------------
    # Test 11: Moment Conditions Properties
    #--------------------------------------------------------------------------
    @testset "Moment Conditions" begin
        Random.seed!(888)
        N = 200
        J = 3
        K = 2
        X = [ones(N) randn(N)]
        Y, _ = sim_logit_with_gumbel(N, J)
        
        α_true = randn(K * (J-1))
        
        # GMM objective should be non-negative
        obj_true = mlogit_gmm(α_true, X, Y)
        obj_wrong = mlogit_gmm(10.0 * randn(K * (J-1)), X, Y)
        
        @test obj_true >= 0
        @test obj_wrong >= 0
    end
    
    #--------------------------------------------------------------------------
    # Test 12: Robustness to Sample Size
    #--------------------------------------------------------------------------
    @testset "Robustness to Sample Size" begin
        Random.seed!(999)
        J = 3
        
        for N in [50, 100, 500, 1000]
            Y, X = sim_logit_with_gumbel(N, J)
            
            @test length(Y) == N
            @test size(X, 1) == N
            @test all(1 .<= Y .<= J)
            
            # Should have representation of all choices (with some tolerance for small N)
            if N >= 100
                @test length(unique(Y)) == J
            end
        end
    end
    
    #--------------------------------------------------------------------------
    # Test 13: SMM Function
    #--------------------------------------------------------------------------
    @testset "SMM Objective Function" begin
        Random.seed!(1111)
        N = 100
        J = 3
        K = 2
        X = [ones(N) randn(N)]
        y = rand(1:J, N)
        α = randn(K * (J-1))
        D = 10  # Small number of simulations for testing
        
        # Test that function runs without error
        @test isa(mlogit_smm_overid(α, X, y, D), Number)
        @test mlogit_smm_overid(α, X, y, D) ≥ 0
        @test !isnan(mlogit_smm_overid(α, X, y, D))
        @test !isinf(mlogit_smm_overid(α, X, y, D))
    end
    
    #--------------------------------------------------------------------------
    # Test 14: Comparison of Estimation Methods
    #--------------------------------------------------------------------------
    @testset "Comparison of Estimation Methods" begin
        Random.seed!(1212)
        N = 1000
        J = 3
        
        # Simulate data with known parameters
        Y, X = sim_logit_with_gumbel(N, J)
        
        K = size(X, 2)
        α_init = zeros(K * (J-1))
        
        # Test MLE
        result_mle = optimize(α -> mlogit_mle(α, X, Y),
                             α_init,
                             LBFGS(),
                             Optim.Options(iterations=500))
        @test Optim.converged(result_mle)
        
        # Test GMM
        result_gmm = optimize(α -> mlogit_gmm(α, X, Y),
                             α_init,
                             LBFGS(),
                             Optim.Options(iterations=500))
        @test Optim.converged(result_gmm)
        
        # Estimates should be similar
        @test norm(result_mle.minimizer - result_gmm.minimizer) < 1.0
    end
    
    #--------------------------------------------------------------------------
    # Test 15: Input Validation
    #--------------------------------------------------------------------------
    @testset "Input Validation" begin
        Random.seed!(1313)
        N = 50
        J = 3
        K = 2
        X = [ones(N) randn(N)]
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        # Test that functions handle edge cases
        @test isa(mlogit_mle(α, X, y), Number)
        @test isa(mlogit_gmm(α, X, y), Number)
        @test isa(mlogit_gmm_overid(α, X, y), Number)
        
        # Test with single observation (edge case)
        X_single = reshape([1.0, 0.5], 1, 2)
        y_single = [1]
        @test isa(mlogit_mle(α, X_single, y_single), Number)
    end
end

println("\n" * "="^80)
println("All tests completed!")
println("="^80)

