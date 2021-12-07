
function driven_equilibrium(
	α::AbstractVector{<: Real},
	ϕ::AbstractVector{<: Real},
	TR::Union{Tuple{Real, Real}, AbstractVector{<: Real}},
	kmax::Integer,
	cycles::Integer,
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	R1::Real,
	R2::Real;
	record::Union{Val{:signal}, Val{:all}} = Val(:signal)
)

	# Only support TR = Tuple
	TR, TW = TR # (W for waiting)

	# Precompute inter-cycle relaxation
	# G is zero, and τ is the waiting time
	combined_relaxation_inter_cycle = compute_combined_relaxation(TW, kmax, [0.0], [TW], D, R1, R2)

	# TODO: Allow record all, need generated function as well

	# First cycle
	signal, state = simulate(
		Val(:full_out),
		α, ϕ, TR,
		kmax,
		G, τ, D,
		R1, R2;
		record
	)

	# All other cycles
	error = Vector{Float64}(undef, cycles-1)
	for cycle = 2:cycles
		last_signal = copy(signal)
		apply_combined_relaxation!(state, kmax+1, combined_relaxation_inter_cycle...)	
		signal, state = simulate(
			Val(:full),
			α, ϕ, TR,
			kmax,
			G, τ, D,
			R1, R2;
			initial_state=state,
			record
		)
		error[cycle-1] = norm(last_signal .- signal)
	end
	return signal, error
end

