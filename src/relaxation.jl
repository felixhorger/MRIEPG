
include("PartialGrid.jl")

# 1=longitudinal, 2=transverse

@generated function prepare_relaxation(
	TR::Union{Real, AbstractVector{<: Real}},
	R::Union{NTuple{2, <: Real}, PartialGrid{<: Real}},
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	kmax::Integer
)
	if R <: NTuple{2, Real}
		check_R = quote
			@assert R[1] > 0 && R[2] > 0
			num_systems = 1
		end
	else
		check_R = quote
			@assert all(R.q1 .> 0) && all(R.q2 .> 0)
			num_systems = R.num_systems
		end
	end

	return quote
		# Check arguments
		all(TR .> 0)
		$check_R
		# G, τ, kmax checked in diffusion_b_values below

		# Compute the diffusion weights
		D1, D2 = diffusion_b_values(G, τ, kmax) # Misuse of names, these are actually b-values
		@. D1 = diffusion_factor(D1, D) # Now the names are correct, these are diffusion damping factors
		@. D2 = diffusion_factor(D2, D) # for the phase graph

		return compute_relaxation!(TR, R, D1, D2), num_systems
	end
end




# Time-constant TR and scalar R1, R2
# Can precompute everything
struct ConstantRelaxation
	one_minus_E1::Float64
	ED1::Vector{Float64} # Joined relaxation and diffusion, axes goes along k direction
	ED2::Vector{Float64}
end
supported_kmax(relaxation::ConstantRelaxation) = length(relaxation.ED1) - 1

function compute_relaxation!( # Bang because Di are used and not copied (they remain unchanged)
	TR::Real,
	R::NTuple{2, <: Real}, # Required because PartialGrid can be used with MultiSystemRelaxation
	D1::AbstractVector{<: Real},
	D2::AbstractVector{<: Real}
)::ConstantRelaxation
	# Compute combined effect of T1, T2 and diffusion
	E1 = exp(-TR * R[1])
	E2 = exp(-TR * R[2])
	@. D1 = E1 * D1
	@. D2 = E2 * D2
	return ConstantRelaxation(1.0 - E1, D1, D2)
end
@inline function apply_relaxation!(
	state::AbstractMatrix{ComplexF64},
	relaxation::ConstantRelaxation,
	upper::Int64,
	_::Int64,
	_::Int64
)
	# Fixed types because used in functions defined here, so no generic types required
	@inbounds begin
		for (s, ED) in enumerate((relaxation.ED2, relaxation.ED2, relaxation.ED1))
			for i = 1:upper
				state[i, s] *= ED[i]
			end
		end
		state[1, 3] += relaxation.one_minus_E1
	end
	return
end



# Scalar R1, R2 and Vector valued TR
# Can precompute exp(-TR * Ri) and Di, but not their combination
struct TimeDependentRelaxation
	E1::Vector{Float64} # Axes goes along time direction
	E2::Vector{Float64}
	D1::Vector{Float64} # Axes goes along k direction
	D2::Vector{Float64}
end
supported_kmax(relaxation::TimeDependentRelaxation) = length(relaxation.D1) - 1

function compute_relaxation!(
	TR::AbstractVector{<: Real}, 
	R::NTuple{2, <: Real}, # Required because PartialGrid can be used with MultiSystemRelaxation
	D1::AbstractVector{<: Real},
	D2::AbstractVector{<: Real}
)::TimeDependentRelaxation
	# Compute exponentials for T1, T2 relaxation factors
	E1 = @. exp(-TR * R[1])
	E2 = @. exp(-TR * R[2])

	return TimeDependentRelaxation(E1, E2, D1, D2)
end
@inline function apply_relaxation!(
	state::Matrix{ComplexF64},
	relaxation::TimeDependentRelaxation,
	upper::Int64,
	_::Int64,
	t::Int64
)
	@inbounds @views let
		E1 = relaxation.E1[t] 
		E2 = relaxation.E2[t] 
		# Apparently this does not allocate memory for the views ... but haven't checked in this scenario,
		# only REPL sandbox
		@. state[1:upper, 1] *= E2 * relaxation.D2[1:upper]
		@. state[1:upper, 2] *= E2 * relaxation.D2[1:upper]
		@. state[1:upper, 3] *= E1 * relaxation.D1[1:upper]
		state[1, 3] += 1.0 - E1
	end
	return
end



macro iterate_partial_grid(partial_grid, loop_body_a, loop_body_b)
	# In each iteration, the respective array elements are in i:j
	# loop_body_a: do everything independent of q1, since q2 is constant over i:j
	# To get the respective q2 = partial_grid.q2[n]
	# variable names: 
	#	- index n in the array partial_grid.q2
	# loop_body_b: q1 varies (variable name q1)
	# To get the respective q1 = $partial_grid.q1[m]
	# variable names:
	#	- index m in the array partial_grid.q1
	#	- index l corresponding index in i:j
	esc(quote
		local i = 1
		for (n, δ) = enumerate($partial_grid.Δ)
			j = i + δ - 1
			$loop_body_a
			for (m, l) = enumerate(i:j)
				$loop_body_b
			end
			i += δ
		end
	end)
end

# Macro for not duplicating the algorithm for traversing through systems
macro partial_grid_relaxation(partial_grid, compute_E1, longitudinal_scalar, transverse_scalar)
	# These arguments are a bit handwavy
	esc(quote
		@inbounds @views begin
			# Iterate systems and k (system index changes fastest, then k)
			for k = 1:upper # k not in physical units
				# Get diffusion weights
				D1 = relaxation.D1[k]
				D2 = relaxation.D2[k]
				# Offset in array due to k
				k_offset = (k - 1) * relaxation.E.num_systems
				@iterate_partial_grid(
					$partial_grid,
					state[i+k_offset : j+k_offset, 1:2] .*= $transverse_scalar * D2, # Transverse relaxation
					begin # Longitudinal relaxation
						$compute_E1
						state[l + k_offset, 3] *= $longitudinal_scalar * D1
					end
				)
			end

			# Second part of longitudinal relaxation
			@iterate_partial_grid(
				$partial_grid,
				nothing, # Nothing where q2 is involved
				state[l, 3] += 1 - $longitudinal_scalar # Add term to Z(k = 0) state
			)
		end
	end)
end



# Scalar TR and vector valued R1, R2
# Can precompute exp(-TR * Ri) and Di, but not their combination
# but it will be used to distinguish time and graph axes
# (the latter if multiple EPGs are simulated at once, i.e. vector values R1,2).
struct MultiSystemRelaxation
	E::PartialGrid{Float64} # Axes go along system direction (aligned with k direction)
	D1::Vector{Float64} # Axes go along k direction
	D2::Vector{Float64}
end
supported_kmax(relaxation::MultiSystemRelaxation) = length(relaxation.D1) - 1

function compute_relaxation!(
	TR::Real,
	R::PartialGrid{<: Real}, # Relaxivities
	D1::AbstractVector{<: Real},
	D2::AbstractVector{<: Real}
)
	# Compute exponentials for T1, T2 relaxation factors
	E = PartialGrid(
		(@. exp(-TR * R.q1)), # E1
		(@. exp(-TR * R.q2)), # E2
		R.Δ
	)
	return MultiSystemRelaxation(E, D1, D2)
end
function apply_relaxation!(
	state::Matrix{ComplexF64},
	relaxation::MultiSystemRelaxation,
	upper::Int64,
	kmax::Int64,
	_::Int64
)
	@partial_grid_relaxation(
		relaxation.E, # partial_grid: contains relaxation factors exp(-TR * Ri)
		nothing, # longitudinal_E1: already precomputed
		relaxation.E.q1[m], # longitudinal_scalar: get E1
		relaxation.E.q2[n] # transverse_scalar: get E2
	)
	return
end



# Vector valued TR, R1, R2
# Can only precompute diffusion, otherwise memory explodes
struct MultiSystemMultiTRRelaxation
	TR::Vector{Float64} # Axis goes along time direction
	R::PartialGrid{Float64} # Axes go along system direction (aligned with k direction)
	D1::Vector{Float64} # Axes go along k direction
	D2::Vector{Float64}
end
supported_kmax(relaxation::MultiSystemMultiTRRelaxation) = length(relaxation.D1) - 1

function compute_relaxation!(
	TR::AbstractVector{<: Real}, 
	R::PartialGrid{<: Real}, # Relaxivities
	D1::AbstractVector{<: Real},
	D2::AbstractVector{<: Real}
)::MultiSystemMultiTRRelaxation
	return MultiSystemMultiTRRelaxation(TR, R, D1, D2)
end
@inline function apply_relaxation!(
	state::Matrix{ComplexF64},
	relaxation::MultiSystemMultiTRRelaxation,
	upper::Int64,
	kmax::Int64,
	t::Int64
)
	error("Precompute PartialGrids for each TR")
	@partial_grid_relaxation(
		R, # partial_grid: relaxivities Ri
		E1 = exp(-relaxation.TR[t] * R.q1[m]), # longitudinal_E1: precompute scalar
		E1, # longitudinal_scalar
		exp(-relaxation.TR[t] * R.q2[n]) # transverse_scalar
	)
	return
end




