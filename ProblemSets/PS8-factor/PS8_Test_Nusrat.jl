using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, MultivariateStats, FreqTables, ForwardDiff, LineSearches

cd(@__DIR__)

include("PS8_Source_Nusrat.jl")


@testset "PS8 Factor Models Tests" begin
    
    # Load test data once
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
    df = load_data(url)
    
    @testset "Question 1: Load Data and Base Regression" begin
        @test size(df, 1) > 0
        @test size(df, 2) == 13
        @test all(in.([:logwage, :black, :hispanic, :female, :schoolt, :gradHS, :grad4yr], Ref(names(df))))
        
        # Test base regression
        result = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
        @test isa(result, StatsModels.TableRegressionModel)
        @test length(coef(result)) == 7  # 6 variables + intercept
        @test !any(isnan.(coef(result)))
    end
    
    @testset "Question 2: ASVAB Correlations" begin
        cordf = compute_asvab_correlations(df)
        
        @test size(cordf) == (6, 6)
        @test all(-1 .<= Matrix(cordf) .<= 1)
        @test all(diag(Matrix(cordf)) .≈ 1.0)  # Diagonal should be 1
        @test issymmetric(Matrix(cordf))  # Correlation matrix is symmetric
        @test all(Matrix(cordf) .> 0)  # All correlations should be positive for ASVAB
    end
    
    @testset "Question 3: Full Regression with ASVAB" begin
        result = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + 
                            asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), df)
        
        @test isa(result, StatsModels.TableRegressionModel)
        @test length(coef(result)) == 13  # 12 variables + intercept
        @test !any(isnan.(coef(result)))
        @test r2(result) > 0
    end
    
    @testset "Question 4: PCA Generation" begin
        df_pca = copy(df)
        df_pca = generate_pca!(df_pca)
        
        @test :asvabPCA in names(df_pca)
        @test size(df_pca, 2) == size(df, 2) + 1
        @test !any(ismissing.(df_pca.asvabPCA))
        @test !any(isnan.(df_pca.asvabPCA))
        
        # Test PCA regression
        result = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabPCA), df_pca)
        @test length(coef(result)) == 8
        @test !any(isnan.(coef(result)))
    end
    
    @testset "Question 5: Factor Analysis" begin
        df_fa = copy(df)
        df_fa = generate_factor!(df_fa)
        
        @test :asvabFactor in names(df_fa)
        @test size(df_fa, 2) == size(df, 2) + 1
        @test !any(ismissing.(df_fa.asvabFactor))
        @test !any(isnan.(df_fa.asvabFactor))
        
        # Test FA regression
        result = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabFactor), df_fa)
        @test length(coef(result)) == 8
        @test !any(isnan.(coef(result)))
    end
    
    @testset "Question 6: Factor Model Preparation" begin
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        # Check dimensions
        @test size(X, 1) == size(df, 1)
        @test size(X, 2) == 7  # 6 covariates + constant
        @test length(y) == size(df, 1)
        @test size(Xfac, 1) == size(df, 1)
        @test size(Xfac, 2) == 4  # 3 demographics + constant
        @test size(asvabs) == (size(df, 1), 6)
        
        # Check for missing values
        @test !any(isnan.(X))
        @test !any(isnan.(y))
        @test !any(isnan.(Xfac))
        @test !any(isnan.(asvabs))
        
        # Check constant columns
        @test all(X[:, end] .== 1.0)
        @test all(Xfac[:, end] .== 1.0)
    end
    
    @testset "Question 6: Factor Model Likelihood" begin
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        # Create reasonable starting values
        K = size(X, 2)
        L = size(Xfac, 2)
        J = size(asvabs, 2)
        
        θ_test = vcat(
            vec(zeros(L, J)),  # γ
            zeros(K),           # β
            0.1 * ones(J+1),   # α
            0.5 * ones(J+1)    # σ
        )
        
        # Test likelihood function runs
        ll = factor_model(θ_test, X, Xfac, asvabs, y, 5)
        @test !isnan(ll)
        @test !isinf(ll)
        @test ll isa Real
        
        # Test with different quadrature points
        ll_9 = factor_model(θ_test, X, Xfac, asvabs, y, 9)
        @test !isnan(ll_9)
        @test !isinf(ll_9)
    end
    
    @testset "Dimension Consistency" begin
        # Test that all ASVAB columns exist
        asvab_cols = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
        @test all(in.(asvab_cols, Ref(names(df))))
        
        # Test that dimensions match across operations
        df_test = copy(df)
        df_test = generate_pca!(df_test)
        df_test = generate_factor!(df_test)
        
        @test size(df_test, 1) == size(df, 1)
        @test :asvabPCA in names(df_test)
        @test :asvabFactor in names(df_test)
    end
    
    @testset "Numerical Stability" begin
        # Test that functions handle edge cases
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        # Test with small but valid parameters
        K = size(X, 2)
        L = size(Xfac, 2)
        J = size(asvabs, 2)
        
        θ_small = vcat(
            1e-4 * ones(L*J),
            1e-4 * ones(K),
            1e-4 * ones(J+1),
            1e-1 * ones(J+1)
        )
        
        ll_small = factor_model(θ_small, X, Xfac, asvabs, y, 5)
        @test !isnan(ll_small)
        @test !isinf(ll_small)
    end
end