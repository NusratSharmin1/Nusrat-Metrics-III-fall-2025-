using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

# Set working directory and include source
cd(@__DIR__)

include("PS5_Source_Nusrat.jl")


@testset "PS5 Bus Engine Replacement Tests" begin
    
    @testset "Data Loading Tests" begin
        @test_nowarn df_long = load_static_data()
        
        df_long = load_static_data()
        @test isa(df_long, DataFrame)
        @test all(in.([:bus_id, :time, :Y, :Odometer, :RouteUsage, :Branded], Ref(names(df_long))))
        @test nrow(df_long) > 0
        @test all(df_long.Y .∈ Ref([0, 1]))  # Binary decision
        
        @test_nowarn d = load_dynamic_data()
        
        d = load_dynamic_data()
        @test isa(d, NamedTuple)
        @test haskey(d, :Y) && haskey(d, :X) && haskey(d, :N) && haskey(d, :T)
        @test size(d.Y) == (d.N, d.T)
        @test size(d.X) == (d.N, d.T)
        @test length(d.B) == d.N
        @test d.β == 0.9
    end
    
    @testset "Future Value Computation Tests" begin
        # Create minimal test data
        d_test = (
            zbin = 2, xbin = 3, T = 2, β = 0.9,
            xval = [0.0, 0.5, 1.0],
            xtran = [0.8 0.2 0.0; 0.6 0.3 0.1; 0.4 0.4 0.2; 0.8 0.2 0.0; 0.6 0.3 0.1; 0.4 0.4 0.2]
        )
        
        θ_test = [1.0, -0.1, 0.5]
        FV = zeros(d_test.zbin * d_test.xbin, 2, d_test.T + 1)
        
        @test_nowarn compute_future_value!(FV, θ_test, d_test)
        
        # Test terminal condition
        @test all(FV[:, :, d_test.T + 1] .== 0.0)
        
        # Test that FV values are finite
        @test all(isfinite.(FV[:, :, 1:d_test.T]))
        
        # Test that FV decreases over time (generally expected)
        @test all(FV[:, :, 1] .>= FV[:, :, 2])
    end
    
    @testset "Log Likelihood Tests" begin
        # Load real data for likelihood test
        d = load_dynamic_data()
        
        # Test with reasonable parameter values
        θ_test = [2.0, -0.15, 1.0]
        
        @test_nowarn ll = log_likelihood_dynamic(θ_test, d)
        
        ll = log_likelihood_dynamic(θ_test, d)
        @test isa(ll, Real)
        @test isfinite(ll)
        @test ll > 0  # Should be positive since we return -loglike
        
        # Test that likelihood changes with different parameters
        θ_test2 = [1.5, -0.1, 0.8]
        ll2 = log_likelihood_dynamic(θ_test2, d)
        @test ll2 != ll
    end
    
    @testset "Static Model Tests" begin
        df_long = load_static_data()
        
        @test_nowarn model = estimate_static_model(df_long)
        
        model = estimate_static_model(df_long)
        @test isa(model, GLM.GeneralizedLinearModel)
        @test length(coef(model)) == 3  # Intercept + Odometer + Branded
    end
    
    @testset "Integration Test" begin
        # Test that main() runs without errors (but don't run full optimization)
        @test_nowarn begin
            # Load data
            df_long = load_static_data()
            d = load_dynamic_data()
            
            # Test one likelihood evaluation
            θ_test = [2.0, -0.15, 1.0]
            ll = log_likelihood_dynamic(θ_test, d)
            
            # Verify it's reasonable
            @test isfinite(ll) && ll > 0
        end
    end
end
