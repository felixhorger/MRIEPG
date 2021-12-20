
function driven_equilibrium(
	cycles::Integer,
	kmax::Integer,
	α::AbstractVector{<: Real},
	ϕ::AbstractVector{<: Real},
	TR::Union{Tuple{Real, Real}, AbstractVector{<: Real}},
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	R::Union{NTuple{2, <: Real}, PartialGrid{Float64}},
	record::Union{Val{:signal}, Val{:all}} = Val(:signal)
)
	# TODO: Allow record all, need generated function as well

	# Only support TR = Tuple
	TR, TW = TR # (W for waiting)

	# Precompute inter-cycle relaxation
	# G is zero, and τ is the waiting time
	relaxation_inter_cycle, _ = prepare_relaxation(TW, R, [0.0], Float64[TW], D, kmax)

	# First cycle
	signal, state = simulate(Val(:full_out), kmax, α, ϕ, TR, G, τ, D, R, nothing, record)

	# All other cycles
	error = Vector{Float64}(undef, cycles-1)
	for cycle = 2:cycles
		last_signal = copy(signal)
		#apply_relaxation!(state, relaxation_inter_cycle, kmax+1, kmax, 0)	
		signal, state = simulate(
			Val(:full),
			kmax,
			α, ϕ, TR,
			G, τ, D,
			R,
			state,
			record
		)
		error[cycle-1] = norm(last_signal .- signal)
	end
	return signal, error
end

