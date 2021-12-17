
include("PartialGrid.jl")

# 1=longitudinal, 2=transverse

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
				state[i, s] *= ED[mod1(i, kmax+1)]
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




# Macro for not duplicating the algorithm for traversing through systems
macro iterate_partial_grid(preamble, partial_grid, longitudinal_E1, longitudinal_scalar, transverse_scalar)
	# These arguments are a bit handwavy
	esc(quote
		@inbounds @views begin
			# Extract information from relaxation struct
			$preamble
			D1 = relaxation.D1[1:upper]
			D2 = relaxation.D2[1:upper]

			# Iterate systems (sorted by relaxivities)
			# i,j refer systems (different relaxivities), but each system has K states (k = 0...kmax)
			local i = 1
			K = kmax + 1
			for (n, δ) = enumerate($partial_grid.Δ)
				j = i + δ - 1
				# Transverse relaxation
				Falsch: brauche nicht upper, sondern num_systems, da zuerst das system sich aendert, dann k
				folglich nehme scalar von D_i, und array von transverse scalar und longitudinal scalar
				for l = i:j
					lk = (l-1) * K + 1 # One based indexing mumble mumble ...
					state[lk:(lk + upper - 1), 1:2] .*= $transverse_scalar .* D2
				end
				# Longitudinal relaxation
				for (m, l) = enumerate(i:j)
					lk = (l-1) * K + 1
					$longitudinal_E1
					state[lk:(lk + upper - 1), 3] .*= $longitudinal_scalar .* D1
					state[lk, 3] += 1 - $longitudinal_scalar
				end
				i += δ
			end
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
	@iterate_partial_grid(
		E = relaxation.E, # preamble: extract from relaxation struct, unsure whether this actually makes a difference
		E, # partial_grid: contains relaxation factors exp(-TR * Ri)
		nothing, # longitudinal_E1: already precomputed
		E.q1[m], # longitudinal_scalar: get E1
		E.q2[n] # transverse_scalar: get E2
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
	@iterate_partial_grid(
		TR = relaxation.TR[t], # preamble, get value of TR
		R, # partial_grid: relaxivities Ri
		E1 = exp(-TR * R.q1[m]), # longitudinal_E1: precompute scalar
		E1, # longitudinal_scalar
		exp(-TR * R.q2[n]) # transverse_scalar
	)
	return
end




