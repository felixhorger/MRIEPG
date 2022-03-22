
module MRIEPG
	
	using LinearAlgebra
	using MRIConst: γ # Used in diffusion.jl
	using XLargerYs

	include("kmax.jl")
	include("memory.jl")
	include("rf_pulses.jl")
	include("diffusion.jl")
	include("relaxation.jl")
	include("simulate.jl")
	include("driven_equilibrium.jl")

end

