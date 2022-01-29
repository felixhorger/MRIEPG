
module MRIEPG
	
	using LinearAlgebra
	using MRIConst: γ

	include("kmax.jl")
	include("memory.jl")
	include("rf_pulses.jl")
	include("diffusion.jl")
	include("relaxation.jl")
	include("simulate.jl")
	include("driven_equilibrium.jl")

	# TODO: function to revert phase and return real signal arrays
end

