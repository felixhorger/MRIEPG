
function driven_equilibrium(
	cycles::Integer,
	α::AbstractVector{<: Real},
	ϕ::AbstractVector{<: Real},
	TR::Tuple{Real, Real},
	kmax::Integer,
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	R::Union{NTuple{2, <: Real}, TriangularGrid{Float64}}
)
	# TODO: Allow record all, then add another function to compute the error between cycles
	# record signal, last, lastall (epg of last cycle), all (all epgs)

	TR, TW = TR # (W for waiting)

	# Precompute inter-cycle relaxation
	# G is zero, and τ is the waiting time
	relaxation_inter_cycle, _ = prepare_relaxation(TW, R, [0.0], Float64[TW], D, kmax)

	# First cycle
	signal, state = simulate(Val(:full_out), kmax, α, ϕ, TR, G, τ, D, R, nothing, record)

	# All other cycles
	#error = Vector{Float64}(undef, cycles-1)
	for cycle = 2:cycles
		last_signal = copy(signal)

		apply_relaxation!(state, relaxation_inter_cycle, kmax+1, kmax, 0)

		signal, state = simulate(
			Val(:full),
			kmax,
			α, ϕ, TR,
			G, τ, D,
			R,
			state,
			record
		)
		#error[cycle-1] = norm(last_signal .- signal)
	end
	return signal, error
end


function driven_equilibrium!(
	cycles::Integer,
	rf_matrices::AbstractArray{<: Complex, 3},
	inter_cycle_relaxation::Union{ConstantRelaxation, MultiSystemRelaxation},
	relaxation::Union{ConstantRelaxation, MultiTRRelaxation, MultiSystemRelaxation, MultiSystemMultiTRRelaxation},
	num_systems::Integer,
	kmax::Integer,
	memory::SimulationMemory
)
	# TODO: See the version for constant TR = TW
	error(" # TODO: See the version for constant TR = TW")
	# TODO: Allow for recording all, maybe make argument with array of recordings, default all nothings

	timepoints = size(rf_matrices, 3)
	final_state_index = get_final_state(timepoints)

	# First cycle
	driven_equilibrium_cycle!(
		Val(:full_out),
		rf_matrices,
		inter_cycle_relaxation,
		relaxation,
		final_state_index,
		kmax,
		memory
	)

	# All other cycles
	for cycle = 2:cycles-1
		# Reshuffle states
		reorder_states(
			memory, final_state_index,
			memory.recording # Keep original recording
		)
		driven_equilibrium_cycle!(
			Val(:full),
			kmax,
			rf_matrices,
			inter_cycle_relaxation, relaxation,
			memory, final_state_index
		)
	end
	# Reshuffle states and use proper recording for the last cycle
	reorder_states(
		memory, final_state_index,
		allocate_recording(Val(:signal), timepoints, num_systems, 0)
	)
	# Simulate last cycle
	driven_equilibrium_cycle!(
		Val(:full_in),
		kmax,
		rf_matrices,
		inter_cycle_relaxation, relaxation,
		memory, final_state_index
	)
	return permutedims(memory.recording, (2,1))
end


@generated function driven_equilibrium!(
	cycles::Integer,
	rf_matrices::AbstractArray{<: Complex, 3},
	relaxation::Union{ConstantRelaxation, MultiTRRelaxation, MultiSystemRelaxation, MultiSystemMultiTRRelaxation},
	num_systems::Integer,
	kmax::Integer,
	memory::SimulationMemory,
	out::AbstractArray{ComplexF64, N}
) where N

	if N == 2
		!(out <: Array{ComplexF64, 2}) && error("For the case N = 2 views do not work")
		set_recording = :nothing
		final_recording = :( out )
	elseif N > 2
		indices = Vector{Symbol}(undef, N+1)
		indices[1:end-1] .= :(:)
		indices[end] = :(cycle)
		set_recording = :( recording[$(indices...)] .= memory.recording )
		final_recording = :( memory.recording )
	else
		error("N must be in (2, 3, 4)")
	end

	return quote
		timepoints = size(rf_matrices, 3)
		final_state_index = get_final_state(timepoints)

		# First cycle
		simulate!(
			Val(:full_out), kmax,
			rf_matrices,
			relaxation, num_systems,
			memory
		)
		cycle = 1
		$set_recording

		# All other cycles
		for cycle = 2:cycles-1
			# Reshuffle states
			memory = reorder_states(
				memory, final_state_index,
				memory.recording
			)
			simulate!(
				Val(:full), kmax,
				rf_matrices,
				relaxation, num_systems,
				memory
			)
			$set_recording
		end
		# Reshuffle states and use proper recording for the last cycle
		memory = reorder_states(
			memory, final_state_index,
			$final_recording
		)
		# Simulate last cycle
		simulate!(
			Val(:full_in), kmax,
			rf_matrices,
			relaxation, num_systems,
			memory
		)
		cycle = cycles
		$set_recording
		return memory.recording
	end
end

@inline function reorder_states(
	memory::SimulationMemory,
	final_state_index::Integer,
	recording::Union{Matrix{ComplexF64}, Array{ComplexF64, 3}, Nothing}
)
	memory = SimulationMemory(
		# Only switch pointers, here C would be useful ... mumble mumble
		(
			memory.two_states[3-final_state_index],
			memory.two_states[final_state_index]
		),
		recording
	)
end

@inline function driven_equilibrium_cycle!(
	mode::Union{Val{:full}, Val{:full_in}, Val{:full_out}},
	kmax::Integer,
	rf_matrices::AbstractArray{<: Complex, 3},
	inter_cycle_relaxation::Union{ConstantRelaxation, MultiSystemRelaxation},
	relaxation::Union{ConstantRelaxation, MultiTRRelaxation, MultiSystemRelaxation, MultiSystemMultiTRRelaxation},
	memory::SimulationMemory,
	final_state_index::Integer
)
	# Single cycle
	simulate!(mode, kmax, rf_matrices, relaxation, memory)
	# Inter cycle relaxation
	apply_relaxation!(
		memory.two_states[final_state_index],
		relaxation_inter_cycle,
		kmax+1, # upper
		kmax,
		0 # t, dummy
	)
	return
end

