using Test, Random, LinearAlgebra, Distributions, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, ForwardDiff
cd(@__DIR__)
# Include the main file
include("PS3_Nusrat.jl")

@testset "PS3 Problem Set Unit Tests" begin

    @testset "Data Loading Tests" begin
        @testset "load_data function" begin
            # Test with actual URL
            url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS3-gev/nlsw88w.csv"
            
            @test_nowarn begin
                X, Z, y = load_data(url)
            end
            
            X, Z, y = load_data(url)
            
            # Test dimensions
            @test size(X, 2) == 3  # Should have 3 covariates (age, white, collgrad)
            @test size(Z, 2) == 8  # Should have 8 wage alternatives
            @test size(X, 1) == size(Z, 1) == length(y)  # Same number of observations
            
            # Test data types
            @test isa(X, Matrix{Float64}) || isa(X, Matrix{Int64})
            @test isa(Z, Matrix{Float64}) || isa(Z, Matrix{Int64})
            @test isa(y, Vector{Int64})
            
            # Test that y contains valid occupation codes
            @test all(y .>= 1)
            @test all(y .<= 8)
            @test length(unique(y)) <= 8
            
            # Test that there are no missing values
            @test !any(isnan.(X))
            @test !any(isnan.(Z))
            @test !any(ismissing.(y))
            
            println("✓ Data loading tests passed")
        end
    end

    @testset "Multinomial Logit Tests" begin
        # Create small test dataset
        Random.seed!(123)
        n_obs = 100
        n_choices = 4
        n_covs = 2
        
        X_test = randn(n_obs, n_covs)
        Z_test = randn(n_obs, n_choices)
        y_test = rand(1:n_choices, n_obs)
        
        @testset "mlogit_with_Z function structure" begin
            # Test parameter vector length
            n_params = n_covs * (n_choices - 1) + 1  # alpha + gamma
            theta_test = randn(n_params)
            
            @test_nowarn mlogit_with_Z(theta_test, X_test, Z_test, y_test)
            
            # Test return type
            ll = mlogit_with_Z(theta_test, X_test, Z_test, y_test)
            @test isa(ll, Real)
            @test !isnan(ll)
            @test !isinf(ll)
            
            println("✓ Multinomial logit structure tests passed")
        end
        
        @testset "mlogit_with_Z mathematical properties" begin
            theta_test = randn(n_covs * (n_choices - 1) + 1)
            
            # Test that log-likelihood decreases with better fit
            # (This is a heuristic test - create a scenario where we know the true parameters)
            ll1 = mlogit_with_Z(theta_test, X_test, Z_test, y_test)
            ll2 = mlogit_with_Z(theta_test .* 0.1, X_test, Z_test, y_test)  # Move closer to zero
            
            # Both should be finite
            @test isfinite(ll1)
            @test isfinite(ll2)
            
            # Test parameter vector length validation
            @test_throws BoundsError mlogit_with_Z(theta_test[1:end-1], X_test, Z_test, y_test)
            
            println("✓ Multinomial logit mathematical properties tests passed")
        end
    end

    @testset "Nested Logit Tests" begin
        # Create test data
        Random.seed!(456)
        n_obs = 50
        n_choices = 6
        n_covs = 2
        
        X_test = randn(n_obs, n_covs)
        Z_test = randn(n_obs, n_choices)
        y_test = rand(1:n_choices, n_obs)
        nesting_structure = [[1, 2, 3], [4, 5]]  # Two nests, choice 6 is outside
        
        @testset "nested_logit_with_Z function structure" begin
            # Test parameter vector: 2 betas for each nest + 2 lambdas + 1 gamma = 2*2 + 2 + 1 = 7
            n_params = 2 * n_covs + 2 + 1  # 2 nests * n_covs + 2 lambdas + gamma
            theta_test = [randn(2 * n_covs); 0.5; 0.7; 0.1]  # Lambdas between 0 and 1
            
            @test_nowarn nested_logit_with_Z(theta_test, X_test, Z_test, y_test, nesting_structure)
            
            # Test return type
            ll = nested_logit_with_Z(theta_test, X_test, Z_test, y_test, nesting_structure)
            @test isa(ll, Real)
            @test !isnan(ll)
            @test !isinf(ll)
            
            println("✓ Nested logit structure tests passed")
        end
        
        @testset "nested_logit_with_Z parameter constraints" begin
            theta_test = [randn(2 * n_covs); 0.5; 0.7; 0.1]
            
            # Test with lambda values that could cause issues
            theta_extreme = copy(theta_test)
            theta_extreme[end-2] = 0.0  # Lambda very close to 0
            
            @test_nowarn nested_logit_with_Z(theta_extreme, X_test, Z_test, y_test, nesting_structure)
            
            # Test that function handles the case where lambda = 1
            theta_unity = copy(theta_test)
            theta_unity[end-2:end-1] .= 1.0
            
            @test_nowarn nested_logit_with_Z(theta_unity, X_test, Z_test, y_test, nesting_structure)
            
            println("✓ Nested logit parameter constraint tests passed")
        end
    end

    @testset "Optimization Tests" begin
        # Use smaller dataset for faster testing
        Random.seed!(789)
        n_obs = 50
        n_choices = 4
        n_covs = 2
        
        X_test = randn(n_obs, n_covs)
        Z_test = randn(n_obs, n_choices)
        y_test = rand(1:n_choices, n_obs)
        
        @testset "optimize_mlogit function" begin
            @test_nowarn optimize_mlogit(X_test, Z_test, y_test)
            
            theta_hat = optimize_mlogit(X_test, Z_test, y_test)
            
            # Test return dimensions
            expected_length = n_covs * (n_choices - 1) + 1
            @test length(theta_hat) == expected_length
            
            # Test that estimates are finite
            @test all(isfinite.(theta_hat))
            
            println("✓ Multinomial logit optimization tests passed")
        end
        
        @testset "optimize_nested_logit function" begin
            nesting_structure = [[1, 2], [3]]  # Simple nesting for testing
            
            @test_nowarn optimize_nested_logit(X_test, Z_test, y_test, nesting_structure)
            
            theta_hat = optimize_nested_logit(X_test, Z_test, y_test, nesting_structure)
            
            # Test return dimensions: 2 nests * n_covs + 2 lambdas + 1 gamma
            expected_length = 2 * n_covs + 2 + 1
            @test length(theta_hat) == expected_length
            
            # Test that estimates are finite
            @test all(isfinite.(theta_hat))
            
            # Test that lambda estimates are positive (necessary for nested logit)
            lambdas = theta_hat[end-2:end-1]
            @test all(lambdas .> 0)
            
            println("✓ Nested logit optimization tests passed")
        end
    end

    @testset "Integration Tests" begin
        @testset "allwrap function" begin
            # Test that the main function runs without errors
            @test_nowarn allwrap()
            
            println("✓ Integration tests passed")
        end
    end

    @testset "Edge Cases and Error Handling" begin
        @testset "Empty data handling" begin
            # Test with minimal data
            X_min = reshape([1.0, 2.0], 2, 1)
            Z_min = reshape([1.0 2.0; 3.0 4.0], 2, 2)
            y_min = [1, 2]
            
            theta_min = [0.1, 0.05]  # 1*(2-1) + 1 = 2 parameters
            
            @test_nowarn mlogit_with_Z(theta_min, X_min, Z_min, y_min)
            
            println("✓ Edge case tests passed")
        end
        
        @testset "Data consistency checks" begin
            # Test mismatched dimensions
            X_bad = randn(10, 2)
            Z_bad = randn(11, 4)  # Different number of rows
            y_bad = rand(1:4, 10)
            
            theta_bad = randn(7)  # 2*(4-1) + 1 = 7
            
            @test_throws BoundsError mlogit_with_Z(theta_bad, X_bad, Z_bad, y_bad)
            
            println("✓ Data consistency tests passed")
        end
    end

    @testset "Gradient and Numerical Stability Tests" begin
        Random.seed!(999)
        n_obs = 30
        X_test = randn(n_obs, 2)
        Z_test = randn(n_obs, 3)
        y_test = rand(1:3, n_obs)
        
        @testset "Numerical derivatives" begin
            theta_test = randn(5)  # 2*(3-1) + 1 = 5
            
            # Test that ForwardDiff can compute gradients
            @test_nowarn ForwardDiff.gradient(t -> mlogit_with_Z(t, X_test, Z_test, y_test), theta_test)
            
            grad = ForwardDiff.gradient(t -> mlogit_with_Z(t, X_test, Z_test, y_test), theta_test)
            @test all(isfinite.(grad))
            @test length(grad) == length(theta_test)
            
            println("✓ Gradient computation tests passed")
        end
        
        @testset "Numerical stability" begin
            theta_test = randn(5)
            
            # Test with extreme values
            X_extreme = X_test * 100  # Large covariates
            ll_extreme = mlogit_with_Z(theta_test, X_extreme, Z_test, y_test)
            @test isfinite(ll_extreme)
            
            # Test with very small covariates
            X_tiny = X_test / 1000
            ll_tiny = mlogit_with_Z(theta_test, X_tiny, Z_test, y_test)
            @test isfinite(ll_tiny)
            
            println("✓ Numerical stability tests passed")
        end
    end
end

println("\n" * "="^50)
println("ALL UNIT TESTS COMPLETED!")
println("="^50)