
module MRIEPG
	
	using LinearAlgebra
	using MRIConst: γ

	include("kmax.jl")
	include("rf_pulse_matrix.jl")
	include("diffusion.jl")
	include("combined_relaxation.jl")
	include("simulate.jl")
	include("driven_equilibrium.jl")

	# TODO preparation and simulate_kernel used in simulate and driven_equilibrium
	# kernel takes preallocated state arrays
	# TODO: function to revert phase and return real signal arrays

end

