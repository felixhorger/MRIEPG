
function check_arguments(
	α::AbstractVector{<: Real},
	ϕ::AbstractVector{<: Real},
	TR::Real,
	kmax::Integer
)
	timepoints = length(α)
	@assert length(ϕ) == timepoints
	@assert TR > 0
	@assert kmax >= 0
	return timepoints
end

function check_arguments(
	α::AbstractVector{<: Real},
	ϕ::AbstractVector{<: Real},
	TR::AbstractVector{<: Real},
	kmax::Integer
)
	timepoints = length(α)
	@assert length(ϕ) == timepoints
	@assert length(TR) == timepoints && all(TR .> 0)
	@assert kmax >= 0
	return timepoints
end



# Arrays holding the current phase graph state
# but indices start at 1, while k starts at zero
# First index for k, second index for components, basically F^+(k), F^-(-k), Z(k) stacked
# Use two allocated memory blocks
function allocate_states(
	mode::Union{Val{:minimal}, Val{:full}, Val{:full_in}, Val{:full_out}},
	initial_state::Nothing,
	kmax::Integer
)::Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}}
	allocated_states = (
		zeros(ComplexF64, kmax+1, 3), # +1 for k = 0
		zeros(ComplexF64, kmax+1, 3) # Needs to be zero, otherwise ugly
	)
	allocated_states[2][1, 3] = 1 # Set M0, start with second block
	return allocated_states
end

function allocate_states(
	mode::Union{Val{:full}, Val{:full_in}},
	initial_state::AbstractMatrix{<: Complex},
	kmax::Integer
)::Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}}
	@assert size(initial_state) == (kmax+1, 3) # +1 for k = 0
	copy_of_initial_state = Matrix{ComplexF64}(undef, size(initial_state))
	copy_of_initial_state .= initial_state
	return Matrix{ComplexF64}(undef, size(initial_state)), copy_of_initial_state
end



@inline function allocate_recording(record::Val{:signal}, timepoints::Int64, kmax::Int64)
	Vector{ComplexF64}(undef, timepoints)
end
@inline function allocate_recording(record::Val{:all}, timepoints::Int64, kmax::Int64)
	Array{ComplexF64}(undef, kmax+1, 3, timepoints)
end



@inline function record_state(
	record::Val{:signal},
	recorded::Vector{ComplexF64},
	state::Matrix{ComplexF64},
	t::Integer
)
	@inbounds recorded[t] = state[1, 2] # Record F^-(k = 0)
end
@inline function record_state(
	record::Val{:all},
	recorded::Array{ComplexF64, 3},
	state::Matrix{ComplexF64},
	t::Integer
)
	@inbounds recorded[:, :, t] = state
end



"""
	α, ϕ in radians
	TR in the unit determined by R1,2
	kmax in indices
	G in mT/m
	τ in ms
	D in m^2 / time unit matching TR (magnitude 10^-12)

	(TODO: More complicated sequences with different gradient moments)
"""
@generated function simulate(
	mode::Union{Val{:minimal}, Val{:full}, Val{:full_in}, Val{:full_out}},
	α::AbstractVector{<: Real},
	ϕ::AbstractVector{<: Real},
	TR::Union{Real, AbstractVector{<: Real}},
	kmax::Integer,
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	R1::Real,
	R2::Real;
	initial_state::Union{Nothing, AbstractMatrix{<: Complex}} = nothing,
	record::Union{Val{:signal}, Val{:all}} = Val(:signal)
)
	# Select what to return: full final state or recording only
	# Also, select when to finish the loop
	if mode <: Val{:full_out} || mode <: Val{:full}
		return_value = :((recorded, state))
	else
		return_value = :recorded
	end

	# Body of the function in quote
	return quote
		# Check arguments
		timepoints = check_arguments(α, ϕ, TR, kmax)

		# Pre-allocate memory for the current and the next phase graph state
		allocated_states = allocate_states(mode, initial_state, kmax)

		# Precompute relaxation (T1, T2 and diffusion)
		combined_relaxation = compute_combined_relaxation(TR, kmax, G, τ, D, R1, R2)

		# Allocate memory for recording EPG states in time
		recorded = allocate_recording(record, timepoints, kmax)

		# Allocate memory for the pulse matrix
		T = Matrix{ComplexF64}(undef, 3, 3)

		# Simulate
		local state
		for t = 1:timepoints
			# How many states are involved?
			upper = required_states(mode, t, timepoints, kmax)

			# Apply pulse
			@inbounds let
				rf_pulse_matrix(α[t], ϕ[t]; out=T)
				j = mod1(t, 2)
				state = allocated_states[j]
				@views mul!(state[1:upper, :], allocated_states[3-j][1:upper, :], T)
			end

			# Store signal (F^-(k = 0)) or complete state
			record_state(record, recorded, state, t)
			#=
				Note:
				No need to apply any relaxation before getting the signal, because it is only a scaling.
				Of course, this only holds as long as the echo time is constant.
				Also, relaxation does not have to be computed in the last iteration if
				the final state is not returned.
				However, in that case upper equals one which means the following is more or less for free.
			=#

			# Apply relaxation (T1, T2 and diffusion)
			apply_combined_relaxation!(state, combined_relaxation, upper, t)

			# Shift (by gradients)
			@inbounds let
				# F^+(k) goes to F^+(k+1)
				for i = min(upper+1, kmax+1):-1:2
					state[i, 1] = state[i-1, 1]
				end
				# F^+(k = 0) becomes the conjugate of F^-(k = -1)
				state[1, 1] = conj(state[2, 2])
				# F^-(k) also goes to F^-(k+1), but k is ordered the other way around in the array
				for i = 1:(upper-1)
					state[i, 2] = state[i+1, 2]
				end
				# F^-(k = -upper+1) becomes unpopulated
				state[upper, 2] = 0
				# Longitudinal states are not getting shifted
			end
		end
		return $return_value
	end
end
# TODO: Need off resonance version. Maybe use "nothing" if no off-res should be simulated and Float if it should.
# Check if initial state and mode are compatible

