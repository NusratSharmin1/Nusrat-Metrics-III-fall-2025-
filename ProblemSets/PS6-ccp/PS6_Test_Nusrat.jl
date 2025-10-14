using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

cd(@__DIR__)

 include("PS6_Source_Nusrat.jl")

@testset "PS6 Rust Model CCP Estimation Tests" begin
    
    Random.seed!(12345)
    
    @testset "Data Loading and Reshaping Tests" begin
        
        @testset "load_and_reshape_data basic functionality" begin
            # Test with actual data URL
            url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
            
            df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
            
            # Test output structure
            @test isa(df_long, DataFrame)
            @test :bus_id in names(df_long)
            @test :time in names(df_long)
            @test :Y in names(df_long)
            @test :Odometer in names(df_long)
            @test :Xstate in names(df_long)
            @test :Zst in names(df_long)
            
            # Test dimensions
            @test size(Xstate, 2) == 20  # 20 time periods
            @test length(Zstate) == size(Xstate, 1)  # Same number of buses
            @test length(Branded) == size(Xstate, 1)
            
            # Test that time variable is correctly created
            @test minimum(df_long.time) == 1
            @test maximum(df_long.time) == 20
            
            # Test that Y is binary
            @test all(x -> x in [0, 1], df_long.Y)
        end
        
        @testset "data consistency checks" begin
            url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
            df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
            
            # Check that each bus appears 20 times
            bus_counts = combine(groupby(df_long, :bus_id), nrow => :count)
            @test all(bus_counts.count .== 20)
            
            # Check sorting
            @test issorted(df_long, [:bus_id, :time])
        end
    end
    
    @testset "Flexible Logit Estimation Tests" begin
        
        @testset "estimate_flexible_logit basic functionality" begin
            # Create mock data
            n_obs = 100
            mock_df = DataFrame(
                Y = rand([0, 1], n_obs),
                Odometer = rand(50000:200000, n_obs),
                RouteUsage = rand(1:5, n_obs),
                Branded = rand([0, 1], n_obs),
                time = rand(1:20, n_obs)
            )
            
            model = estimate_flexible_logit(mock_df)
            
            @test isa(model, GeneralizedLinearModel)
            @test model.model.rr.d isa Binomial
            @test isa(model.model.rr.l, LogitLink)
            
            # Test predictions are between 0 and 1
            predictions = predict(model, mock_df)
            @test all(0 .<= predictions .<= 1)
        end
    end
    
    @testset "State Space Construction Tests" begin
        
        @testset "construct_state_space dimensions" begin
            # Create test grids
            xbin, zbin = 5, 3
            xval = [1.0, 2.0, 3.0, 4.0, 5.0]
            zval = [1.0, 2.0, 3.0]
            xtran = rand(xbin * zbin, xbin)  # Mock transition matrix
            
            state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
            
            @test nrow(state_df) == xbin * zbin
            @test ncol(state_df) == 4  # Odometer, RouteUsage, Branded, time
            @test :Odometer in names(state_df)
            @test :RouteUsage in names(state_df)
            @test :Branded in names(state_df)
            @test :time in names(state_df)
        end
        
        @testset "state space value ranges" begin
            xbin, zbin = 3, 2
            xval = [10.0, 20.0, 30.0]
            zval = [1.0, 2.0]
            xtran = rand(xbin * zbin, xbin)
            
            state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
            
            @test Set(state_df.Odometer) == Set(xval)
            @test Set(state_df.RouteUsage) == Set(zval)
            @test all(state_df.Branded .== 0)
            @test all(state_df.time .== 0)
        end
    end
    
    @testset "Future Value Computation Tests" begin
        
        @testset "compute_future_values dimensions" begin
            # Setup test parameters
            xbin, zbin, T = 3, 2, 5
            β = 0.9
            
            # Create mock state_df and model
            state_df = DataFrame(
                Odometer = repeat([1.0, 2.0, 3.0], zbin),
                RouteUsage = repeat([1.0, 2.0], inner=xbin),
                Branded = zeros(xbin * zbin),
                time = zeros(xbin * zbin)
            )
            
            # Create mock model that returns reasonable predictions
            mock_predictions = rand(0.1:0.01:0.9, xbin * zbin)
            mock_model = (state_df) -> mock_predictions
            
            # Mock the predict function for testing
            function mock_predict(model, df)
                return rand(0.1:0.01:0.9, nrow(df))
            end
            
            # Temporarily replace predict function
            original_predict = predict
            Base.eval(Main, :(predict(model, df) = $(mock_predict)(model, df)))
            
            try
                FV = compute_future_values(state_df, mock_model, rand(xbin*zbin, xbin), 
                                         xbin, zbin, T, β)
                
                @test size(FV) == (xbin * zbin, 2, T + 1)
                @test all(FV[:, :, 1] .== 0)  # Initial period should be zero
                @test all(FV[:, :, 2:end] .> 0)  # Future values should be positive
            finally
                # Restore original predict function
                Base.eval(Main, :(predict = $original_predict))
            end
        end
    end
    
    @testset "FVT1 Computation Tests" begin
        
        @testset "compute_fvt1 output length" begin
            # Create mock inputs
            N, T, xbin = 5, 3, 4
            
            mock_df_long = DataFrame(
                bus_id = repeat(1:N, inner=T),
                time = repeat(1:T, N),
                Y = rand([0, 1], N * T)
            )
            
            FV = rand(xbin * 2, 2, T + 1)  # zbin = 2 for this test
            xtran = rand(xbin * 2, xbin)
            Xstate = rand(1:xbin, N, T)
            Zstate = rand(1:2, N)
            Branded = rand([0, 1], N)
            
            fvt1_long = compute_fvt1(mock_df_long, FV, xtran, Xstate, Zstate, xbin, Branded)
            
            @test length(fvt1_long) == N * T
            @test isa(fvt1_long, Vector{Float64})
        end
    end
    
    @testset "Structural Parameter Estimation Tests" begin
        
        @testset "estimate_structural_params basic functionality" begin
            # Create mock data
            n_obs = 200
            mock_df = DataFrame(
                Y = rand([0, 1], n_obs),
                Odometer = rand(50000:200000, n_obs),
                Branded = rand([0, 1], n_obs)
            )
            
            fvt1 = rand(-2.0:0.1:2.0, n_obs)
            
            model = estimate_structural_params(mock_df, fvt1)
            
            @test isa(model, GeneralizedLinearModel)
            @test model.model.rr.d isa Binomial
            @test isa(model.model.rr.l, LogitLink)
            
            # Check that offset was used
            @test :fv in names(mock_df)
            
            # Check coefficient names
            coef_names = coefnames(model)
            @test "(Intercept)" in coef_names
            @test "Odometer" in coef_names
            @test "Branded" in coef_names
        end
    end
    
    @testset "Grid Creation Integration Tests" begin
        
        @testset "create_grids output structure" begin
            zval, zbin, xval, xbin, xtran = create_grids()
            
            @test isa(zval, Vector)
            @test isa(xval, Vector)
            @test isa(xtran, Matrix)
            @test zbin == length(zval)
            @test xbin == length(xval)
            @test size(xtran, 1) == xbin * zbin
            @test size(xtran, 2) == xbin
            
            # Transition probabilities should sum to 1
            @test all(abs.(sum(xtran, dims=2) .- 1) .< 1e-10)
        end
    end
    
    @testset "Integration Tests" begin
        
        @testset "main function runs without error" begin
            # Test that main function can run (may take time with real data)
            # This is more of a smoke test
            @test_nowarn begin
                # Redirect output to avoid cluttering test results
                original_stdout = stdout
                (rd, wr) = redirect_stdout()
                
                try
                    main()
                catch e
                    # Restore stdout first
                    redirect_stdout(original_stdout)
                    close(wr)
                    close(rd)
                    rethrow(e)
                end
                
                redirect_stdout(original_stdout)
                close(wr)
                close(rd)
            end
        end
    end
    
    @testset "Edge Cases and Error Handling" begin
        
        @testset "empty data handling" begin
            empty_df = DataFrame(
                Y = Int[],
                Odometer = Float64[],
                RouteUsage = Float64[],
                Branded = Int[],
                time = Int[]
            )
            
            # Should handle empty data gracefully or throw appropriate error
            @test_throws Exception estimate_flexible_logit(empty_df)
        end
        
        @testset "parameter bounds" begin
            # Test that discount factor is reasonable
            β = 0.9
            @test 0 < β < 1
            
            # Test that state values are positive
            zval, zbin, xval, xbin, xtran = create_grids()
            @test all(xval .> 0)
            @test all(zval .> 0)
        end
    end
end

