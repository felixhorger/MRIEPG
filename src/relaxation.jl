
# 1=longitudinal, 2=transverse
# Time-constant TR and scalar R1, R2
# Can precompute everything
struct ConstantRelaxation
	one_minus_E1::Float64
	ED1::Vector{Float64} # Joined relaxation and diffusion, axes goes along k direction
	ED2::Vector{Float64}
end
supported_kmax(relaxation::ConstantRelaxation) = length(relaxation.ED1) - 1
function check_relaxation(relaxation::ConstantRelaxation, timepoints::Integer, kmax::Integer)
	if any(length.((relaxation.ED1, relaxation.ED2)) .<= kmax)
		error("kmax of the relaxation data structure is too small.")
	end
	return
end
function compute_relaxation(
	TR::Real,
	R::NTuple{2, <: Real}, # Required because XLargerY can be used with MultiSystemRelaxation
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	kmax::Integer
)::Tuple{ConstantRelaxation, Int64}
	@assert TR > 0
	@assert sum(τ) == TR
	# Compute combined effect of T1, T2 and diffusion
	E1 = exp(-TR * R[1])
	E2 = exp(-TR * R[2])
	b_longitudinal, b_transverse = diffusion_b_values(G, τ, kmax)
	ED1 = (@. b_longitudinal = E1 * diffusion_factor(b_longitudinal, D))
	ED2 = (@. b_transverse = E2 * diffusion_factor(b_transverse, D))
	return ConstantRelaxation(1 - E1, ED1, ED2), 1
end
@inline @inbounds @views function apply_relaxation!( # TODO: inline?
	target_state::AbstractMatrix{ComplexF64},
	source_state::AbstractMatrix{ComplexF64},
	relaxation::ConstantRelaxation,
	upper::Int64,
	_::Int64,
	_::Int64
)
	for (s, ED) in enumerate((relaxation.ED2, relaxation.ED2, relaxation.ED1))
		@. target_state[1:upper, s] = source_state[1:upper, s] * ED[1:upper]
	end

	return
end

@inline @inbounds function apply_longitudinal_recovery!(
	state::AbstractMatrix{ComplexF64},
	relaxation::ConstantRelaxation,
	upper::Int64,
	_::Int64,
	_::Int64
)
	state[1, 3] += relaxation.one_minus_E1

	return
end



"""
	Assumes that the gradients are the same in every TR, so only the "dead time" afterwards changes
"""
function compute_multiTR_diffusion(
	TR::AbstractVector{<: Real},
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	kmax::Real
)
	timepoints = length(TR)
	b_longitudinal_per_time, b_transverse_contribution = prepare_diffusion_b_values(G, τ, kmax)
	D1 = Matrix{Float64}(undef, kmax+1, timepoints)
	D2 = Matrix{Float64}(undef, kmax+1, timepoints)
	@inbounds @views for t = 1:timepoints
		@. D1[:, t] = b_longitudinal_per_time * TR[t]
		@. D2[:, t] = D1[:, t] + b_transverse_contribution
		@. D1[:, t] = diffusion_factor(D1[:, t], D)
		@. D2[:, t] = diffusion_factor(D2[:, t], D)
	end
	return D1, D2
end


# Scalar R1, R2 and Vector valued TR
# Can precompute exp(-TR * Ri) and Di, but not their combination
struct MultiTRRelaxation
	E1::Vector{Float64} # Axes goes along time direction
	E2::Vector{Float64}
	D1::Matrix{Float64} # Axes goes along k direction
	D2::Matrix{Float64}
end
supported_kmax(relaxation::MultiTRRelaxation) = size(relaxation.D1, 1) - 1
function check_relaxation(relaxation::MultiTRRelaxation, timepoints::Integer, kmax::Integer)
	if any(
		(
			length.((relaxation.E1, relaxation.E2))...,
			size.((relaxation.D1, relaxation.D2), 2)...
		) .!= timepoints
	)
		error("Relaxation data structure does not have a matching time axis.")
	end
	if any(size.((relaxation.D1, relaxation.D2), 1) .<= kmax)
		error("kmax of the relaxation data structure is too small.")
	end
	return
end
function compute_relaxation(
	TR::AbstractVector{<: Real}, 
	R::NTuple{2, <: Real}, # Required because XLargerY can be used with MultiSystemRelaxation
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	kmax::Integer
)::Tuple{MultiTRRelaxation, Int64}

	# Check arguments
	@assert all(t -> t > 0, TR)
	@assert R[1] > 0 && R[2] > 0
	# G, τ, kmax checked in compute_multiTR_diffusion

	# Diffusion
	D1, D2 = compute_multiTR_diffusion(TR, G, τ, D, kmax)

	# Compute exponentials for T1, T2 relaxation factors
	E1 = @. exp(-TR * R[1])
	E2 = @. exp(-TR * R[2])

	return MultiTRRelaxation(E1, E2, D1, D2), 1 # num_systems
end

@inline @inbounds @views function apply_relaxation!(
	target_state::Matrix{ComplexF64},
	source_state::Matrix{ComplexF64},
	relaxation::MultiTRRelaxation,
	upper::Int64,
	_::Int64,
	t::Int64
)
	E1 = relaxation.E1[t]
	E2 = relaxation.E2[t]
	D1 = relaxation.D1[1:upper, t]
	D2 = relaxation.D2[1:upper, t]
	# Apparently this does not allocate memory for the views ... but haven't checked in this scenario,
	# only REPL sandbox
	@. target_state[1:upper, 1] = source_state[1:upper, 1] * E2 * D2
	@. target_state[1:upper, 2] = source_state[1:upper, 2] * E2 * D2
	@. target_state[1:upper, 3] = source_state[1:upper, 3] * E1 * D1

	return
end


@inline @inbounds function apply_longitudinal_recovery!(
	state::Matrix{ComplexF64}, # TODO: actually, this should be consistent with apply_relaxation!() source target?
	relaxation::MultiTRRelaxation,
	upper::Int64,
	_::Int64,
	t::Int64
)
	state[1, 3] += 1 - relaxation.E1[t]
	return
end



# Scalar TR and vector valued R1, R2
# Can precompute exp(-TR * Ri) and Di, but not their combination
# but it will be used to distinguish time and graph axes
# (the latter if multiple EPGs are simulated at once, i.e. vector values R1,2).
struct MultiSystemRelaxation
	E::XLargerY{Float64} # Axes go along system direction (aligned with k direction)
	D1::Vector{Float64} # Axes go along k direction
	D2::Vector{Float64}
end
supported_kmax(relaxation::MultiSystemRelaxation) = length(relaxation.D1) - 1
function check_relaxation(relaxation::MultiSystemRelaxation, timepoints::Integer, kmax::Integer)
	if any(length.((relaxation.D1, relaxation.D2)) .<= kmax)
		error("kmax of the relaxation data structure is too small.")
	end
	return
end

function compute_relaxation(
	TR::Real,
	R::XLargerY{<: Real}, # Relaxivities
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	kmax::Integer
)::Tuple{MultiSystemRelaxation, Int64}

	# Check arguments
	@assert TR > 0
	@assert R.x[1] > 0 && R.y[1] > 0 # x,y are sorted
	# G, τ, kmax checked in diffusion_b_values below

	# Diffusion
	b_longitudinal, b_transverse = diffusion_b_values(G, τ, kmax)
	D1 = (@. b_longitudinal = diffusion_factor(b_longitudinal, D))
	D2 = (@. b_transverse = diffusion_factor(b_transverse, D))

	# Compute exponentials for T1, T2 relaxation factors
	E = XLargerY(
		(@. exp(-TR * R.x)), # E2, switched because R2 > R1
		(@. exp(-TR * R.y)), # E1
		R.upper
	)

	return MultiSystemRelaxation(E, D1, D2), R.N
end

# Macro for not duplicating the algorithm for traversing through systems
macro multi_system_relaxation_iterate(grid, D1, D2, E1, E2)
	# These arguments are a bit handwavy
	esc(quote
		# Iterate systems and k (system index changes fastest, then k)
		for k = 1:upper # k not in physical units
			# Get diffusion weights
			D1 = $D1
			D2 = $D2
			# Offset in array due to k
			k_offset = (k - 1) * $grid.N
			@views XLargerYs.@iterate(
				$grid,
				# Compute transverse relaxation in outer loop
				(@. target_state[i+k_offset : j+k_offset, 1:2] = source_state[i+k_offset : j+k_offset, 1:2] * $E2 * D2),
				# Compute first part of longitudinal relaxation in inner loop (decrease current value of magnetisation)
				target_state[l + k_offset, 3] *= $E1 * D1
			)
		end
	end)
end


macro multi_system_longitudinal_recovery_iterate(grid, E1)
	# These arguments are a bit handwavy
	esc(quote
		@views XLargerYs.@iterate(
			$grid,
			# Do nothing in outer loop because nothing R2 related happens
			nothing,
			# Relax longitudinal part
			state[l, 3] += 1 - $E1 # Add to Z(k = 0) state
		)
	end)
end

@inline @inbounds function apply_relaxation!(
	target_state::Matrix{ComplexF64},
	source_state::Matrix{ComplexF64},
	relaxation::MultiSystemRelaxation,
	upper::Int64,
	kmax::Int64,
	_::Int64
)
	 @multi_system_relaxation_iterate(
		relaxation.E,
		relaxation.D1[k],
		relaxation.D2[k],
		relaxation.E.y[m], # E1
		relaxation.E.x[n] # E2
	)
	return
end

@inline @inbounds function apply_longitudinal_recovery!(
	state::Matrix{ComplexF64},
	relaxation::MultiSystemRelaxation,
	_::Int64
)
	 @multi_system_longitudinal_recovery_iterate(
		relaxation.E,
		relaxation.E.y[m], # E1
	)
	return
end



# Vector valued TR, R1, R2
# Can only precompute diffusion, otherwise memory explodes
struct MultiSystemMultiTRRelaxation
	E::Vector{XLargerY{Float64}} # Axes go along time
	D1::Matrix{Float64} # Axes go along k direction
	D2::Matrix{Float64}
end
supported_kmax(relaxation::MultiSystemMultiTRRelaxation) = length(relaxation.D1) - 1
function check_relaxation(relaxation::MultiSystemMultiTRRelaxation, timepoints::Integer, kmax::Integer)
	# TODO: I don't think the below is required?
	#if any(
	#	(
	#		length(relaxation.E),
	#		size.((relaxation.D1, relaxation.D2), 2)...
	#	) .!= timepoints
	#)
	#	error("Relaxation data structure does not have a matching time axis.")
	#end
	if any(size.((relaxation.D1, relaxation.D2), 1) .<= kmax)
		error("kmax of the relaxation data structure is too small.")
	end
	return
end

function compute_relaxation(
	TR::AbstractVector{<: Real}, 
	R::XLargerY{<: Real}, # Relaxivities
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	kmax::Integer
)
	@assert all(t -> t > 0, TR)
	@assert R.x[1] > 0 && R.y[1] > 0 # x,y are sorted
	# G, τ, kmax checked in compute_multiTR_diffusion below

	# Diffusion
	D1, D2 = compute_multiTR_diffusion(TR, G, τ, D, kmax)

	# Compute exponentials for T1, T2 relaxation factors
	timepoints = length(TR)
	E = Vector{XLargerY{Float64}}(undef, timepoints)
	 for t = 1:timepoints
		E[t] = XLargerY(
			(@. exp(-TR[t] * R.x)), # E2, switched because R2 > R1
			(@. exp(-TR[t] * R.y)), # E1
			R.upper
		)
	end
	return MultiSystemMultiTRRelaxation(E, D1, D2), R.N
end

@inline @inbounds function apply_relaxation!(
	target_state::Matrix{ComplexF64},
	source_state::Matrix{ComplexF64},
	relaxation::MultiSystemMultiTRRelaxation,
	upper::Int64,
	kmax::Int64,
	t::Int64
)
	@multi_system_relaxation_iterate(
		relaxation.E[t],
		relaxation.D1[k, t],
		relaxation.D2[k, t],
		relaxation.E[t].y[m], # E1
		relaxation.E[t].x[n] # E2
	)

	return
end

@inline @inbounds function apply_longitudinal_recovery!(
	state::Matrix{ComplexF64},
	relaxation::MultiSystemMultiTRRelaxation,
	t::Int64
)
	@multi_system_longitudinal_recovery_iterate(
		relaxation.E[t],
		relaxation.E[t].y[m], # E1
	)

	return
end

