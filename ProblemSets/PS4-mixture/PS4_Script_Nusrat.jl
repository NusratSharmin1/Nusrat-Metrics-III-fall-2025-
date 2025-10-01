using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions
cd(@__DIR__)

# Set seed for reproducibility
Random.seed!(1234)

# Include the main functions 
include("PS4_Source_Nusrat.jl")

allwrap()


