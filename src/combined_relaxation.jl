
#=
Types for distinguishing the two cases of scalar and vector-valued repetition time TR
In the scalar case, the relaxation happening in TR is constant and varies only with k.
In the vector values case, relaxation varies additionally in time.
=#
struct ScalarTRCombinedRelaxation
	one_minus_E1::Float64
	ED_longitudinal::Vector{Float64}
	ED_transverse::Vector{Float64}
end

struct VectorTRCombinedRelaxation
	E1::Vector{Float64}
	E2::Vector{Float64}
	D_longitudinal::Vector{Float64}
	D_transverse::Vector{Float64}
end



function compute_combined_relaxation(
	TR::Real,
	kmax::Integer,
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	R1::Real,
	R2::Real
)::ScalarTRCombinedRelaxation
	# Compute combined effect of T1, T2 and diffusion
	E1 = exp(-TR * R1)
	E2 = exp(-TR * R2)
	ED_longitudinal, ED_transverse = diffusion_b_values(G, τ, kmax) # Misuse of names
	@. ED_longitudinal = E1 * diffusion_factor(ED_longitudinal, D)
	@. ED_transverse = E2 * diffusion_factor(ED_transverse, D)
	return ScalarTRCombinedRelaxation(1.0 - E1, ED_longitudinal, ED_transverse)
end



function compute_combined_relaxation(
	TR::AbstractVector{<: Real}, 
	kmax::Integer,
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	R1::Real,
	R2::Real
)::VectorTRCombinedRelaxation
	# Compute exponentials for T1, T2 relaxation factors
	E1 = exp.(-TR * R1)
	E2 = exp.(-TR * R2)

	# Compute diffusion damping factors (depends on k)
	D_longitudinal, D_transverse = diffusion_b_values(G, τ, kmax) # Misuse of names
	D_longitudinal = diffusion_factor.(D_longitudinal, D)
	D_transverse = diffusion_factor.(D_transverse, D)

	return VectorTRCombinedRelaxation(E1, E2, D_longitudinal, D_transverse)
end



# Fixed types because used in functions defined here, so no generic types required
@inline function apply_combined_relaxation!(
	state::AbstractMatrix{ComplexF64},
	combined_relaxation::ScalarTRCombinedRelaxation,
	upper::Int64,
	unused::Int64
)
	@inbounds let
		for (s, ED) in enumerate((
			combined_relaxation.ED_transverse,
			combined_relaxation.ED_transverse,
			combined_relaxation.ED_longitudinal
		))
			for i = 1:upper
				state[i, s] *= ED[i]
			end
		end
		state[1, 3] += combined_relaxation.one_minus_E1
	end
	return
end

@inline function apply_combined_relaxation!(
	state::Matrix{ComplexF64},
	combined_relaxation::VectorTRCombinedRelaxation,
	upper::Int64,
	t::Int64
)
	@inbounds let
		E1 = combined_relaxation.E1[t] 
		E2 = combined_relaxation.E2[t] 
		# Apparently this does not allocate memory for the views ... but haven't checked in this scenario,
		# only REPL sandbox
		@views state[1:upper, 1] .*= E2 * combined_relaxation.D_transverse[1:upper]
		@views state[1:upper, 2] .*= E2 * combined_relaxation.D_transverse[1:upper]
		@views state[1:upper, 3] .*= E1 * combined_relaxation.D_longitudinal[1:upper]
		state[1, 3] .+= 1.0 - E1
	end
	return
end

