#=
	The convention that variables are lower case is not fully satisfied herein.
	Gradients are denoted G, diffusion coefficient D, repetition time TR, relaxivities R.
=#
# TODO: use mode not for kmax, but for e.g. GRE, TSE, ...
# rename mode to mask or sth
"""
	TODO: reduce phi to phi0 and Delta phi
	If the user has a more complicated scenario, he/she has to use multiple simulate() calls,
	handing the final state over to the next call.
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
	kmax::Integer,
	α::AbstractVector{<: Real},
	ϕ::AbstractVector{<: Real},
	TR::Union{Real, AbstractVector{<: Real}},
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	R::Union{NTuple{2, <: Real}, PartialGrid{Float64}},
	initial_state::Union{Nothing, AbstractMatrix{<: Number}} = nothing,
	record::Union{Val{:signal}, Val{:all}} = Val(:signal)
)
	# Select what to return: full final state or recording only
	# Also, select when to finish the loop
	if mode <: Val{:full_out} || mode <: Val{:full}
		return_value = :((
			recording,
			memory.two_states[get_final_state(memory, timepoints)]
		))
	else
		return_value = :recording
	end

	# Drop singleton dimension or transpose recording after simulation finished
	if record <: Val{:signal}
		reshape_recording = quote
			if num_systems == 1
				recording = copy(dropdims(memory.recording; dims=1))
			else
				recording = copy(transpose(memory.recording))
			end
		end
	else
		reshape_recording = :(recording = copy(memory.recording))
	end

	# Body of the function in quote
	return quote

		# Precompute relaxation
		relaxation, num_systems = prepare_relaxation(TR, R, G, τ, D, kmax)

		# Get number of timepoints to simulate
		timepoints = length(α)

		# Allocate memory for the simulation
		memory = allocate_memory(mode, timepoints, num_systems, kmax, initial_state, record)

		# Simulate
		simulate!(mode, timepoints, kmax, num_systems, α, ϕ, relaxation, memory)

		$reshape_recording
		return $return_value
	end
end
# TODO: Need off resonance version. Maybe use "nothing" if no off-res should be simulated and Float if it should.
# Check if initial state and mode are compatible


@generated function simulate!(
	mode::Union{Val{:minimal}, Val{:full}, Val{:full_in}, Val{:full_out}},
	timepoints::Int64, 
	kmax::Int64, # I think a relaxation struct made with one kmax can be used for every kmax lower than that.
	num_systems::Int64,
	α::Vector{Float64}, # Replace by RF pulse matrices, also use this in driven_equilibrium
	ϕ::Vector{Float64},
	relaxation::Union{
		ConstantRelaxation,
		TimeDependentRelaxation,
		MultiSystemRelaxation,
		MultiSystemMultiTRRelaxation
	},
	memory::SimulationMemory
)
	# TODO: Not sure if @generated is required here, on the other hand this should ensure that the function
	# is compiled for any combination of input types
	# TODO: Make comment that this is optimised for multiple evaluations, thus is makes sense to
	# 1) Precompute rf matrices
	# 2) Choose relaxation size so that cache is optimally used
	# For "playing around" or evaluating for a single R (like fitting) this is still more than fast enough.
	# However, larger scale fitting could be more efficient if pulse matrices change (such as with relB1 as free parameter, in that case don't precompute)

	return quote

		# Alias into SimulationMemory struct
		rf_matrix = memory.rf_matrix
		two_states = memory.two_states
		recording = memory.recording

		# Check arguments
		@assert length(α) == timepoints
		@assert length(ϕ) == timepoints
		total_num_states = num_systems * (kmax+1)
		@assert supported_kmax(relaxation) >= kmax
		@assert all(size.(memory.two_states, 1) .>= total_num_states)
		check_recording_size(recording, timepoints, total_num_states)

		for t = 1:timepoints
			# How many states are involved?
			upper = required_states(mode, t, timepoints, kmax)
			upper_systems = num_systems * upper

			# Apply pulse
			local state
			@inbounds begin
				# Get matrix
				rf_pulse_matrix(α[t], ϕ[t]; out=rf_matrix)
				# Apply it
				let j = mod1(t, 2)
					state = two_states[j]
					@views mul!(state[1:upper_systems, :], two_states[3-j][1:upper_systems, :], rf_matrix)
				end
			end

			# Store signal (F^-(k = 0)) or complete state
			record_state(recording, state, t, num_systems)
			#=
				Note:
				No need to apply any relaxation before getting the signal, because it is only a scaling.
				Of course, this only holds as long as the echo time is constant.
				Also, relaxation does not have to be computed in the last iteration if
				the final state is not returned.
				However, in that case upper equals one which means the following is more or less for free.
			=#

			# Apply relaxation (T1, T2 and diffusion)
			apply_relaxation!(state, relaxation, upper, kmax, t)

			# Shift (by gradients)
			@inbounds @views let
				# F^+(k) goes to F^+(k+1), only for k > 0
				# The first i is the first index of the k-state which is newly populated by the shift
				#println("+") # Excuse the shotgun debug comment clutter
				for i = min(upper_systems, total_num_states-num_systems) + 1 : -num_systems : num_systems+1
					state[i:i+num_systems-1, 1] .= state[i-num_systems : i-1, 1]
					#println("$(i-num_systems : i-1) -> $(i:i+num_systems-1)")
				end
				# F^+(k = 0) becomes the conjugate of F^-(k = -1)
				state[1:num_systems, 1] .= conj.(state[num_systems+1 : 2*num_systems, 2])
				# F^-(k) also goes to F^-(k+1), but k is ordered the other way around in the array
				#println("-")
				for i = 1:num_systems:(upper_systems-num_systems)
					state[i : i+num_systems-1, 2] .= state[i+num_systems : i+2*num_systems-1, 2]
					#println("$(i+num_systems : i+2*num_systems-1) -> $(i : i+num_systems-1)")
				end
				# F^-(k = -upper+1) becomes unpopulated
				state[upper_systems-num_systems+1 : upper_systems, 2] .= 0
				#println("$(upper_systems-num_systems+1 : upper_systems) -> 0")
				# Longitudinal states are not getting shifted
				#sleep(10)
			end
		end
	end
end



