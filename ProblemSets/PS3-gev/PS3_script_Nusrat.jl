using Random, LinearAlgebra, Distributions, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, ForwardDiff

cd(@__DIR__)

# Read in the function
include("PS3_Nusrat.jl")

# Execute the function
allwrap()

# 2. Interpret the estimated coefficient y
# estimated gamma is -0.094
# gamma represents the change in latent utility
# with 1 unit change in the relative E(log wage)
# in occupation j (relative to other)
# people prefer more wage
