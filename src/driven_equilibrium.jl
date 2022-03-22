
function driven_equilibrium(
	cycles::Integer,
	α::AbstractVector{<: Real},
	ϕ::AbstractVector{<: Real},
	TR::Tuple{Real, Real}, # This is quite inflexible, need generated
	kmax::Integer,
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	R::Union{NTuple{2, <: Real}, XLargerY{Float64}}
)
	# TODO: Allow record all, then add another function to compute the error between cycles
	# record signal, last, lastall (epg of last cycle), all (all epgs)

	TR, TW = TR # (W for waiting)

	timepoints = length(α)
	@assert length(ϕ) == timepoints

	rf_matrices = Array{ComplexF64, 3}(undef, 3, 3, timepoints)
	 @views for t = 1:timepoints
		rf_pulse_matrix!(rf_matrices[:, :, t], α[t], ϕ[t])
	end

	# Precompute inter-cycle relaxation
	# G is zero, and τ is the waiting time
	relaxation, num_systems = compute_relaxation(TR, R, G, τ, D, kmax)
	inter_cycle_relaxation, _ = compute_relaxation(TW, R, [0.0], Float64[TW], D, kmax)

	# Allocate memory
	memory = MRIEPG.allocate_memory(Val(:full), timepoints, num_systems, kmax, nothing, Val(:nothing))
	recording = MRIEPG.allocate_recording(Val(:signal), timepoints, num_systems, 0)

	# Run
	signal = driven_equilibrium!(
		cycles,
		rf_matrices,
		relaxation,
		inter_cycle_relaxation,
		num_systems,
		kmax,
		memory, recording
	)
	# TODO: Extract relevant dimensions like in simulate() when setting up generated function
	return signal
end



@generated function driven_equilibrium!(
	cycles::Integer,
	rf_matrices::AbstractArray{<: Complex, 3},
	relaxation::Union{ConstantRelaxation, MultiTRRelaxation, MultiSystemRelaxation, MultiSystemMultiTRRelaxation},
	inter_cycle_relaxation::Union{ConstantRelaxation, MultiSystemRelaxation, Nothing},
	num_systems::Integer,
	kmax::Integer,
	memory::SimulationMemory,
	out::AbstractArray{ComplexF64, N}
) where N
	# Break in between cycles

	if N == 2
		!(out <: Array{ComplexF64, 2}) && error("For the case N = 2 views do not work")
		set_recording = :()
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

	if inter_cycle_relaxation <: Nothing
		apply_inter_cycle_relaxation = :()
	else
		apply_inter_cycle_relaxation = quote
			apply_relaxation!(
				memory.two_states[final_state_index],
				inter_cycle_relaxation,
				kmax+1, kmax, 0
			)
		end
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
			$apply_inter_cycle_relaxation
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
		$apply_inter_cycle_relaxation
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

