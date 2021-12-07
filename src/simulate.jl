
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
	initial_state::Union{Nothing, AbstractMatrix{ComplexF64}} = nothing, # make this into dispatch, maybe easier with planned prepare method
	record::Union{Val{:signal}, Val{:all}} = Val(:signal)
)
	# TODO: Need off resonance version. Maybe use "nothing" if no off-res should be simulated and Float if it should.
	# Check if initial state and mode are compatible
	use_initial_state = mode <: Union{Val{:full}, Val{:full_in}}
	if initial_state <: Nothing 
		@assert !use_initial_state "No initial state provided"
		setup_state = quote
			# Use two allocated memory blocks
			allocated_states = (
				zeros(ComplexF64, kmax+1, 3), # +1 for k = 0
				zeros(ComplexF64, kmax+1, 3)
			)
			allocated_states[2][1, 3] = 1 # Set M0, start with second block
		end
	else
		@assert use_initial_state "Initial state ignored"
		setup_state = quote
			@assert size(initial_state) == (kmax+1, 3) # +1 for k = 0
			# Use two allocated memory blocks
			allocated_states = (
				copy(initial_state),
				copy(initial_state)
			)
		end
	end

	# Distinguish case of scalar respectively vector-valued TR
	if TR <: AbstractVector{<: Real}
		TR_check = :(N == length(TR) && all(TR .> 0))
		combined_relaxation = :((E1, E2, D_longitudinal, D_transverse))
		apply_combined_relaxation = quote
			@inbounds apply_combined_relaxation!(state, E1[t], E2[t], ED_longitudinal, ED_transverse)
		end
	else
		TR_check = :(TR > 0)
		combined_relaxation = :((one_minus_E1, ED_longitudinal, ED_transverse))
		apply_combined_relaxation = quote
			apply_combined_relaxation!(state, upper, one_minus_E1, ED_longitudinal, ED_transverse)
		end
	end

	# Code to apply the rf pulse matrix
	if mode <: Val{:full}
		apply_rf_pulse_matrix = quote
			mul!(state, allocated_states[3-j], T)
		end
	else
		apply_rf_pulse_matrix = quote
			@inbounds @views mul!(state[1:upper, :], allocated_states[3-j][1:upper, :], T)
		end
	end

	# Code to set up variables to record the state
	if record <: Val{:signal}
		setup_recorded_states = :(Vector{ComplexF64}(undef, N))
		record_state = :(@inbounds recorded_states[t] = state[1, 2]) # Record F^-(k = 0)
	elseif record <: Val{:all}
		setup_recorded_states = :(Array{ComplexF64}(undef, kmax+1, 3, N))
		record_state = :(@inbounds recorded_states[:, :, t] = state)
	end

	# Code to return the recorded states (signal or all) and the full final state if desired
	if mode <: Val{:full_out} || mode <: Val{:full}
		return_value = :((recorded_states, state))
	else
		return_value = :(recorded_states)
	end

	# Interpolate into expression and return full code
	return quote

		# Check arguments
		N = length(α)
		@assert N == length(ϕ)
		@assert $TR_check
		@assert kmax >= 0

		# Precompute relaxation (T1, T2 and diffusion)
		$combined_relaxation = compute_combined_relaxation(TR, kmax, G, τ, D, R1, R2)

		# Array holding the current phase graph state
		# but indices start at 1, while k starts at zero
		# First index for k, second index for components, basically F^+(k), F^-(-k), Z(k) stacked
		$setup_state

		# Allocate memory
		recorded_states = $setup_recorded_states
		T = Matrix{ComplexF64}(undef, 3, 3)

		# Simulate
		local state
		for t = 1:N
			# How many states are involved?
			upper = required_states(mode, t, N, kmax)

			# Apply pulse
			rf_pulse_matrix(α[t], ϕ[t]; out=T)
			j = mod1(t, 2)
			state = allocated_states[j]
			$apply_rf_pulse_matrix

			# Store signal (F^-(k = 0)) or complete state
			$record_state
			# Note: No need to apply any relaxation before getting the signal, because it is only a scaling.
			# Of course, this only holds as long as the echo time is constant.

			# TODO: In last iteration, for some modes it can be stopped here, worth it? Maybe reorder for loop?
		
			# Apply relaxation (T1, T2 and diffusion)
			$apply_combined_relaxation

			# Shift (by gradients)
			@inbounds begin
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

