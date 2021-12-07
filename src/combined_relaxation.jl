
function compute_combined_relaxation(
	TR::Real,
	kmax::Integer,
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	R1::Real,
	R2::Real
)
	# Compute combined effect of T1, T2 and diffusion
	E1 = exp(-TR * R1)
	E2 = exp(-TR * R2)
	ED_longitudinal, ED_transverse = diffusion_b_values(G, τ, kmax) # Misuse of names
	@. ED_longitudinal = E1 * diffusion_factor(ED_longitudinal, D)
	@. ED_transverse = E2 * diffusion_factor(ED_transverse, D)
	return 1.0 - E1, ED_longitudinal, ED_transverse
end

function compute_combined_relaxation(
	TR::AbstractVector{<: Real}, 
	kmax::Integer,
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	R1::Real,
	R2::Real
)
	# Compute exponentials for T1, T2 relaxation factors
	E1 = exp.(-TR * R1)
	E2 = exp.(-TR * R2)

	# Compute diffusion damping factors (depends on k)
	D_longitudinal, D_transverse = diffusion_b_values(G, τ, kmax) # Misuse of names
	D_longitudinal = diffusion_factor.(D_longitudinal, D)
	D_transverse = diffusion_factor.(D_transverse, D)

	return E1, E2, D_longitudinal, D_transverse
end

@inline function apply_combined_relaxation!(
	state::AbstractMatrix{<: Complex},
	E1::Real,
	E2::Real,
	D_longitudinal::AbstractVector{<: Real},
	D_transverse::AbstractVector{<: Real}
)
	@inbounds begin
		state[:, 1] .*= E2 * D_transverse
		state[:, 2] .*= E2 * D_transverse
		state[:, 3] .*= E1 * D_longitudinal
		state[1, 3] .+= 1.0 - E1
	end
	return
end

@inline function apply_combined_relaxation!(
	state::AbstractMatrix{<: Complex},
	upper::Integer,
	one_minus_E1::Real,
	ED_longitudinal::AbstractVector{<: Real},
	ED_transverse::AbstractVector{<: Real}
)
	@inbounds begin
		for (s, ED) in enumerate((ED_transverse, ED_transverse, ED_longitudinal))
			for i = 1:upper
				state[i, s] *= ED[i]
			end
		end
		state[1, 3] += one_minus_E1
	end
	return
end

