#=
	The convention that variables are lower case is not fully satisfied herein.
	Gradients are denoted G, diffusion coefficient D, repetition time TR, relaxivities R.
=#
# TODO: use mode not for kmax, but for e.g. GRE, TSE, ...
# rename mode to mask or sth
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
	initial_state::Union{Nothing, AbstractMatrix{<: Complex}} = nothing,
	record::Union{Val{:signal}, Val{:all}} = Val(:signal)
)
	# Select what to return: full final state or recording only
	# Also, select when to finish the loop
	if mode <: Val{:full_out} || mode <: Val{:full}
		return_value = :((
			memory.recording,
			memory.two_states[get_final_state(memory, timepoints)]
		))
	else
		return_value = :(memory.recording)
	end

	# Body of the function in quote
	return quote

		# Precompute relaxation
		relaxation, timepoints, num_systems = prepare_relaxation(TR, R, G, τ, D, kmax)

		# Check lengths of α and ϕ
		if timepoints == 0
			# TR is scalar
			timepoints = length(α)
		end

		# Allocate memory for the simulation
		total_num_states = (kmax+1) * num_systems # +1 for k = 0
		memory = allocate_memory(mode, timepoints, total_num_states, initial_state, record)

		# Simulate
		simulate!(mode, timepoints, kmax, num_systems, α, ϕ, relaxation, memory)
		return $return_value
	end
end
# TODO: Need off resonance version. Maybe use "nothing" if no off-res should be simulated and Float if it should.
# Check if initial state and mode are compatible



"""
	α, ϕ in radians
	TR in the unit determined by R1,2
	kmax in indices
	G in mT/m
	τ in ms
	D in m^2 / time unit matching TR (magnitude 10^-12)

	(TODO: More complicated sequences with different gradient moments)

"""
@generated function prepare_relaxation(
	TR::Union{Real, AbstractVector{<: Real}},
	R::Union{NTuple{2, <: Real}, PartialGrid{<: Real}},
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	kmax::Integer
)
	if TR <: Real
		check_TR = quote
			@assert TR > 0
			timepoints = 0
		end
	else
		check_TR = quote
			@assert all(TR .> 0)
			timepoints = length(TR)
		end
	end

	if R <: NTuple{2, Real}
		check_R = quote
			@assert R[1] > 0 && R[2] > 0
			num_systems = 1
		end
	else
		check_R = quote
			@assert all(R.q1 .> 0) && all(R.q2 .> 0)
			num_systems = length(R)
		end
	end

	return quote
		# Check arguments
		$check_TR
		$check_R
		# G, τ, kmax checked in diffusion_b_values below

		# Compute the diffusion weights
		D1, D2 = diffusion_b_values(G, τ, kmax) # Misuse of names, these are actually b-values
		@. D1 = diffusion_factor(D1, D) # Now the names are correct, these are diffusion damping factors
		@. D2 = diffusion_factor(D2, D) # for the phase graph

		return compute_relaxation!(TR, R, D1, D2), timepoints, num_systems
	end
end




# Prepare a single simulation run
# This can be in struct because it should be hidden to the user, no parameters should be set here
struct SimulationMemory
	rf_matrix::Matrix{ComplexF64}
	two_states::NTuple{2, Matrix{ComplexF64}}
	recording::Array{ComplexF64}
end
@generated function allocate_memory(
	mode::Union{Val{:minimal}, Val{:full}, Val{:full_in}, Val{:full_out}},
	timepoints::Integer,
	total_num_states::Integer,
	initial_state::Union{Nothing, AbstractMatrix{<: Complex}} = nothing,
	record::Union{Val{:signal}, Val{:all}} = Val(:signal)
)::SimulationMemory
	return quote
		# Pre-allocate memory for the current and the next phase graph state
		two_states = allocate_states(mode, initial_state, total_num_states)

		# Allocate memory for recording EPG states in time
		recording = allocate_recording(record, timepoints, total_num_states)

		# Allocate memory for the pulse matrix
		rf_matrix = Matrix{ComplexF64}(undef, 3, 3)

		return SimulationMemory(rf_matrix, two_states, recording)
	end
end



@generated function simulate!(
	mode::Union{Val{:minimal}, Val{:full}, Val{:full_in}, Val{:full_out}},
	timepoints::Int64, 
	kmax::Int64, # I think a relaxation struct made with one kmax can be used for every kmax lower than that.
	num_systems::Int64,
	α::Vector{Float64},
	ϕ::Vector{Float64},
	relaxation::Union{
		ConstantRelaxation,
		TimeDependentRelaxation,
		MultiSystemRelaxation,
		MultiSystemMultiTRRelaxation
	},
	memory::SimulationMemory
)

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

		local state # Reference to the current state
		for t = 1:timepoints
			# How many states are involved?
			upper_k = required_states(mode, t, timepoints, kmax)
			upper = upper_k * num_systems

			# Apply pulse
			@inbounds let
				rf_pulse_matrix(α[t], ϕ[t]; out=rf_matrix)
				j = mod1(t, 2)
				state = two_states[j]
				@views mul!(state[1:upper, :], two_states[3-j][1:upper, :], rf_matrix)
			end

			# Store signal (F^-(k = 0)) or complete state
			record_state(recording, state, t)
			#=
				Note:
				No need to apply any relaxation before getting the signal, because it is only a scaling.
				Of course, this only holds as long as the echo time is constant.
				Also, relaxation does not have to be computed in the last iteration if
				the final state is not returned.
				However, in that case upper equals one which means the following is more or less for free.
			=#

			# Apply relaxation (T1, T2 and diffusion)
			apply_relaxation!(state, relaxation, upper_k, kmax, t)

			# Shift (by gradients)
			@inbounds @views let
				# F^+(k) goes to F^+(k+1), only for k > 0
				for i = min(upper+num_systems, total_num_states): -num_systems : 2*num_systems
					#state[i:i+num_systems-1, 1] .= state[i-num_systems : i-1, 1]
				end
				# F^+(k = 0) becomes the conjugate of F^-(k = -1)
				state[1:num_systems, 1] .= conj.(state[num_systems+1 : 2*num_systems, 2])
				# F^-(k) also goes to F^-(k+1), but k is ordered the other way around in the array
				for i = 1:num_systems:(upper-num_systems)
					state[i : i+num_systems-1, 2] .= state[i+num_systems : i+2*num_systems-1, 2]
				end
				# F^-(k = -upper+1) becomes unpopulated
				state[upper:upper+num_systems-1, 2] .= 0
				# Longitudinal states are not getting shifted
			end
		end
	end
end




@inline function check_recording_size(recording::Vector{ComplexF64}, timepoints::Int64, _::Int64)
	@assert length(recording) == timepoints
end
@inline function check_recording_size(recording::Array{ComplexF64, 3}, timepoints::Int64, total_num_states::Int64)
	@assert size(recording) == (total_num_states, 3, timepoints)
end



# Arrays holding the current phase graph state
# but indices start at 1, while k starts at zero
# First index for k, second index for components, basically F^+(k), F^-(-k), Z(k) stacked
# Since this code is made for computing multiple systems at once, each k occupies as many elements
# as there are systems. So, along the first axis, the system index changes fastest, k slowest.
# Use two allocated memory blocks
function allocate_states(
	mode::Union{Val{:minimal}, Val{:full}, Val{:full_in}, Val{:full_out}},
	initial_state::Nothing,
	total_num_states::Integer # How many signals are computed at the same time, i.e. length of R1,2
)::NTuple{2, Matrix{ComplexF64}}
	two_states = (
		zeros(ComplexF64, total_num_states, 3), # +1 for k = 0
		zeros(ComplexF64, total_num_states, 3) # Needs to be zeros, otherwise looks ugly in plots
	)
	@inbounds two_states[2][1, 3] = 1 # Set M0, start with second block
	return two_states
end

function allocate_states(
	mode::Union{Val{:full}, Val{:full_in}},
	initial_state::AbstractMatrix{<: Complex},
	total_num_states::Integer,
)::NTuple{2, Matrix{ComplexF64}}
	@assert size(initial_state) == (total_num_states, 3)
	copy_of_initial_state = Matrix{ComplexF64}(undef, total_num_states, 3)
	copy_of_initial_state .= initial_state
	return Matrix{ComplexF64}(undef, total_num_states, 3), copy_of_initial_state
end



# Note that number_of_states = (kmax+1) * num_systems where num_systems is the number of signals computed in parallel
@inline function allocate_recording(record::Val{:signal}, timepoints::Int64, _::Int64)
	Vector{ComplexF64}(undef, timepoints)
end
@inline function allocate_recording(record::Val{:all}, timepoints::Int64, number_of_states::Int64)
	Array{ComplexF64}(undef, number_of_states, 3, timepoints)
end



# Function to record the state of the graph, depending on record mode
@inline function record_state(
	recording::Vector{ComplexF64},
	state::Matrix{ComplexF64},
	t::Integer
)
	@inbounds recording[t] = state[1, 2] # Record F^-(k = 0)
end
@inline function record_state(
	recording::Array{ComplexF64, 3},
	state::Matrix{ComplexF64},
	t::Integer
)
	@inbounds recording[:, :, t] = state
end


@inline function get_final_state(memory::SimulationMemory, timepoints::Integer)
	mod1(timepoints, 2)
end

