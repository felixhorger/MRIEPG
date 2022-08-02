
# Prepare a single simulation run
# This can be in struct because it should be hidden to the user, no parameters should be set here
struct SimulationMemory{T1 <: Matrix{<: Complex}, T2 <: Union{Matrix{<: Complex}, Array{<: Complex, 3}}}
	two_states::NTuple{2, T1}
	recording::T2
end

function allocate_memory(
	mode::Union{Val{:minimal}, Val{:full}, Val{:full_in}, Val{:full_out}},
	timepoints::Integer,
	num_systems::Integer,
	kmax::Integer,
	initial_state::Union{Nothing, AbstractMatrix{<: Number}} = nothing,
	record::Union{Val{:signal}, Val{:all}} = Val(:signal)
)
	# How many states are there in total?
	total_num_states = (kmax+1) * num_systems # +1 for k = 0

	# Pre-allocate memory for the current and the next phase graph state
	two_states = allocate_states(mode, initial_state, num_systems, total_num_states)

	# Allocate memory for recording EPG states in time
	recording = allocate_recording(record, timepoints, num_systems, total_num_states)

	return SimulationMemory(two_states, recording)
end


# TODO: These should return booleans and assert should be in caller
@inline function check_recording_size(recording::Matrix{ComplexF64}, timepoints::Int64, _::Int64)
	@assert size(recording, 2) == timepoints
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
	num_systems::Integer,
	total_num_states::Integer # How many signals are computed at the same time, i.e. length of R1,2
)::NTuple{2, Matrix{ComplexF64}}
	two_states = (
		zeros(ComplexF64, total_num_states, 3), # +1 for k = 0
		zeros(ComplexF64, total_num_states, 3) # Needs to be zeros, otherwise looks ugly in plots
	)
	 two_states[1][1:num_systems, 3] .= 1 # Set M0 of source state
	return two_states
end

function allocate_states(
	mode::Union{Val{:full}, Val{:full_in}},
	initial_state::AbstractMatrix{<: Number},
	_::Integer,
	total_num_states::Integer,
)::NTuple{2, Matrix{ComplexF64}}
	@assert size(initial_state) == (total_num_states, 3)
	# Need "complicated" way so that type is ensured
	copy_of_initial_state = Matrix{ComplexF64}(undef, total_num_states, 3)
	copy_of_initial_state .= initial_state
	# Can return uninitialised array, because mode is full_in
	return copy_of_initial_state, Matrix{ComplexF64}(undef, total_num_states, 3)
end



# Note that num_states = (kmax+1) * num_systems where num_systems is the number of signals computed in parallel
@inline function allocate_recording(
	record::Val{:signal},
	timepoints::Integer,
	num_systems::Integer,
	_::Integer
)
	Matrix{ComplexF64}(undef, num_systems, timepoints)
end
@inline function allocate_recording(
	record::Val{:all},
	timepoints::Integer,
	_::Integer,
	total_num_states::Integer
)
	Array{ComplexF64, 3}(undef, total_num_states, 3, timepoints)
end


# Function to record the state of the graph, depending on record mode
@inline function record_state!(
	recording::Matrix{ComplexF64},
	state::Matrix{ComplexF64},
	t::Integer,
	num_systems::Integer
)
	 @views recording[:, t] = state[1:num_systems, 2] # Record F^-(k = 0)
	 return
end
@inline function record_state!(
	recording::Array{ComplexF64, 3},
	state::Matrix{ComplexF64},
	t::Integer,
	_::Integer
)
	 @views recording[:, :, t] = state
	 return
end

"""
Resets source state to thermal equilibrium
"""
@inline function reset_source_state!(memory::T, num_systems::Integer) where T <: SimulationMemory
	memory.two_states[1][:, 1:2] .= 0
	memory.two_states[1][1:num_systems, 3] .= 1
	memory.two_states[1][num_systems+1:end, 3] .= 0
	return
end
@inline function reset_target_state!(memory::T) where T <: SimulationMemory
	memory.two_states[2] .= 0
	return
end

