# TODO: multi dimensional EPGs
#=
	The convention that variables are lower case is not fully satisfied herein.
	Gradients are denoted G, diffusion coefficient D, repetition time TR, relaxivities R.
=#
# TODO: use mode not for kmax, but for e.g. GRE, (T)SE, ... ?
# rename mode to mask or sth
"""
	α, ϕ in radians
	TR in the unit determined by R1,2
	kmax in indices
	G in mT/m
	τ in ms
	D in m^2 / time unit matching TR (magnitude 10^-12)

"""
# TODO: Shuffle arguments, make mode a default arg and put to end, kmax in front of G, R in before kmax.
function simulate(
	α::AbstractVector{<: Real},
	ϕ::AbstractVector{<: Real},
	TR::Union{Real, AbstractVector{<: Real}},
	R::Union{NTuple{2, <: Real}, XLargerY{Float64}},
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	kmax::Integer,
	mode::Union{Val{:minimal}, Val{:full}, Val{:full_in}, Val{:full_out}}=Val{:minimal},
	initial_state::Union{Nothing, AbstractMatrix{<: Number}} = nothing,
	record::Union{Val{:signal}, Val{:all}, Val{:nothing}} = Val(:signal)
)
	# Precompute relaxation
	relaxation, num_systems = compute_relaxation(TR, R, G, τ, D, kmax)

	# Get number of timepoints and check size of ϕ
	timepoints = length(α)
	@assert length(ϕ) == timepoints

	# Precompute pulse matrices
	rf_matrices = Array{ComplexF64, 3}(undef, 3, 3, timepoints)
	@inbounds @views for t = 1:timepoints
		rf_pulse_matrix!(rf_matrices[:, :, t], α[t], ϕ[t])
	end

	# Allocate memory for the simulation
	memory = allocate_memory(mode, timepoints, num_systems, kmax, initial_state, record)

	# Simulate
	simulate!(1, timepoints, rf_matrices, relaxation, num_systems, kmax, mode, memory)

	recording = reshape_recording(R, memory, record)
	return select_return_value(mode, memory, recording)
end

#TODO: Make function that prepares everything and returns a function that simulates for driven equilibrium


"""
Returns memory with reordered source/target state
"""
function simulate!(
	t0::Integer, # One based indexing!
	timepoints::Integer,
	rf_matrices::AbstractArray{<: Complex, 3},
	relaxation::Union{ConstantRelaxation, MultiTRRelaxation, MultiSystemRelaxation, MultiSystemMultiTRRelaxation},
	num_systems::Integer,
	kmax::Integer, # I think a relaxation struct made with one kmax can be used for every kmax lower than that.
	mode::Union{Val{:minimal}, Val{:full}, Val{:full_in}, Val{:full_out}},
	memory::T where T <: SimulationMemory
)
	# TODO: Not sure if @generated is required here, on the other hand this should ensure that the function
	# is compiled for any combination of input types
	# TODO: Make comment that this is optimised for multiple evaluations, thus is makes sense to
	# 1) Choose relaxation size so that cache is optimally used
	# For "playing around" or evaluating for a single R (like fitting) this is still more than fast enough.
	# However, larger scale fitting could be more efficient if pulse matrices change (such as with relB1 as free parameter, in that case don't precompute)
	# For the future, make a function that computes everything on demand, no precomputation

	# memory.two_states[1] needs to contain the initial state and memory.two_state[2] needs to be zeros!


	# Alias into SimulationMemory struct
	two_states = memory.two_states
	recording = memory.recording

	# Check arguments
	@assert size(rf_matrices, 1) == size(rf_matrices, 2) == 3
	Δt = size(rf_matrices, 3)
	check_relaxation(relaxation, Δt, kmax)
	total_num_states = num_systems * (kmax+1)
	@assert all(
		m -> size(m, 1) >= total_num_states,
		memory.two_states
	)
	check_recording_size(recording, timepoints, total_num_states)

	source_state, target_state = memory.two_states
	 for trel = 1:Δt
		t = trel + t0 - 1
		upper = required_states(mode, t, timepoints, kmax)
		simulate!(
			t, trel,
			relaxation, num_systems,
			rf_matrices,
			upper, kmax, mode,
			source_state, target_state,
			recording
		)
		# Swap source and target
		source_state, target_state = target_state, source_state
	end

	# Reorder states in memory struct
	memory = SimulationMemory((source_state, target_state), recording)

	return memory
end

function simulate!(
	t::Integer,
	trel::Integer,
	relaxation::Union{ConstantRelaxation, MultiTRRelaxation, MultiSystemRelaxation, MultiSystemMultiTRRelaxation},
	num_systems::Integer,
	rf_matrices::AbstractArray{<: Complex, 3},
	upper::Integer,
	kmax::Integer, # I think a relaxation struct made with a spcific kmax can be used for every kmax lower than that, wasn't tested though
	mode::Union{Val{:minimal}, Val{:full}, Val{:full_in}, Val{:full_out}},
	source_state::Matrix{<: Complex},
	target_state::Matrix{<: Complex},
	recording::Union{Matrix{<: Complex}, Array{<: Complex, 3}, Nothing}
)
	# No bounds checked!

	# How many states are involved?
	upper_systems = num_systems * upper

	apply_rf_pulse_matrices!(rf_matrices, source_state, target_state, trel, upper_systems)

	# Store signal (F^-(k = 0)) or complete state
	# TODO: this does not consider echo time. However, the effect of TE can
	# be calculated by the user retrospectively.
	record_state!(recording, target_state, t, num_systems)
	#=
		Note:
		No need to apply any relaxation before getting the signal, because it is only a scaling.
		Of course, this only holds as long as the echo time is constant.
		Also, relaxation does not have to be computed in the last iteration if
		the final state is not returned.
		However, in that case upper equals one which means the following is more or less for free.
	=#

	# Apply relaxation (T1, T2 and diffusion)
	apply_relaxation!(target_state, target_state, relaxation, upper, kmax, trel)
	apply_longitudinal_recovery!(target_state, relaxation, trel)
	# Note: trel is used because depends on TR index, not total time since simulation start.

	gradient_shift!(target_state, num_systems, upper_systems)

	return
end


# Drop singleton dimension or transpose recording after simulation finished
@inline function reshape_recording(R::NTuple{2, Float64}, memory::SimulationMemory, record::Val{:signal})
	dropdims(memory.recording; dims=1)
end
@inline function reshape_recording(R::XLargerY{Float64}, memory::SimulationMemory, record::Val{:signal})
	permutedims(memory.recording, (2,1))
end
@inline function reshape_recording(
	R::Union{NTuple{2, Float64}, XLargerY{Float64}},
	memory::SimulationMemory,
	record::Val{:all}
)
	copy(memory.recording)
end
@inline reshape_recording(
	R::Union{NTuple{2, Float64},
	XLargerY{Float64}},
	memory::SimulationMemory,
	record::Val{:nothing}
) = nothing

# Depending on the simulation mode, select the right return value
@inline function select_return_value(mode::Union{Val{:full_out}, Val{:full}}, memory::SimulationMemory, recording)
	(recording, memory.two_states[1])
end
@inline select_return_value(mode::Union{Val{:minimal}, Val{:full_in}}, _::SimulationMemory, recording) = recording

