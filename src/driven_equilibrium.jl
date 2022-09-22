
function driven_equilibrium(
	cycles::Integer,
	α::AbstractVector{<: Real},
	ϕ::AbstractVector{<: Real},
	TR::Union{Real, AbstractVector{<: Real}},
	R::Union{NTuple{2, <: Real}, XLargerY{Float64}},
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	kmax::Integer,
	record::Union{Val{:signal}, Val{:all}}
)
	# Not sure about this:
	# TODO: Allow record all, then add another function to compute the error between cycles
	# record signal, last, lastall (epg of last cycle), all (all epgs)

	timepoints_per_cycle = length(α)
	@assert length(ϕ) == timepoints_per_cycle
	timepoints = cycles * timepoints_per_cycle

	rf_matrices = Array{ComplexF64, 3}(undef, 3, 3, timepoints_per_cycle)
	 @views for t = 1:timepoints_per_cycle
		rf_pulse_matrix!(rf_matrices[:, :, t], α[t], ϕ[t])
	end

	# Precompute relaxation
	relaxation, num_systems = compute_relaxation(TR, R, G, τ, D, kmax)

	# Allocate memory
	memory = allocate_memory(Val(:minimal), timepoints, num_systems, kmax, nothing, record)

	# Define function for simulation
	function run!(cycle::Integer, memory::SimulationMemory)
		t = (cycle-1) * timepoints_per_cycle + 1
		return @views simulate!(
			t,
			timepoints,
			rf_matrices,
			relaxation,
			num_systems,
			kmax, Val(:minimal),
			memory
		)
	end

	# Run
	memory = driven_equilibrium!(cycles, run!, memory, record)
	return memory.recording
end




"""
	simulation func:
	run!(cycle, memory::SimulationMemory)
	needs to return the memory with reordered states, as returned from simulate!()


	Dimension of recording:
	3: (systems, timepoints of all cycles)
	4: (systems, states, timepoints of all cycles)

	Doesn't reset memory in the end!
	Memory reset does not need to be included in run!()
	Take care that initial memory is reset!

"""
function driven_equilibrium!(
	cycles::Integer,
	run!::Function,
	memory::SimulationMemory,
	record::Union{Val{:signal}, Val{:all}}
)
	@assert cycles ≥ 1

	for cycle = 1:cycles
		memory = run!(cycle, memory)
		reset_target_state!(memory)
	end

	return memory
end

"""
	NRMSE
"""
function driven_equilibrium_convergence(
	signal::AbstractVector{<: Number},
	num_time_per_cycle::Integer
)
	num_time = length(signal)
	deviation = Vector{Float64}(undef, num_time ÷ num_time_per_cycle - 1)
	for cycle in 1:length(deviation)
		t0 = (cycle-1) * num_time_per_cycle + 1
		t1 = t0 + num_time_per_cycle - 1
		t2 = t1 + num_time_per_cycle - 1
		Δ = 0.0
		Σ = 0.0
		for t = t0:t1
			prev = signal[t]
			this = signal[t+num_time_per_cycle]
			Δ += abs2(prev - this)
			Σ += abs2(prev + this)
		end
		deviation[cycle] = sqrt(2.0 * Δ / Σ)
	end
	return deviation
end

