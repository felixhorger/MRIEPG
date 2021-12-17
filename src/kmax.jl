
@inline function required_kmax(num_TR::Integer)::Integer
	@assert num_TR >= 0
	(num_TR ÷ 2) - 1
end


# Code for computing which k-states need to be considered
# +1 in the following are due to indices starting at 1
# No argument checks
@inline required_states(mode::Val{:minimal}, t::Integer, timepoints::Integer, kmax::Integer) = min(t, timepoints-t+1, kmax+1)
@inline required_states(mode::Val{:full}, t::Integer, timepoints::Integer, kmax::Integer) = kmax+1
@inline required_states(mode::Val{:full_in}, t::Integer, timepoints::Integer, kmax::Integer) = min(kmax+1, timepoints-t+1)
@inline required_states(mode::Val{:full_out}, t::Integer, timepoints::Integer, kmax::Integer) = min(t, kmax+1)


function approximation_error_sweep_kmax(
	kmax::OrdinalRange{<: Integer, <: Integer},
	signal_length::Integer,
	func::Function
)
	# Make a note that required kmax is not enforced
	# TODO: also with preparation ... thus version for simulate() and driven_equilibrium() must be separate
	# Also allow T1 and T2 to be arrays
	signals = Vector{Vector{ComplexF64}}(undef, length(kmax))
	for (i, κmax) in enumerate(kmax)
		signals[i] = func(κmax)
	end
		required = required_kmax(signal_length)
		if kmax.stop >= required 
			reference = signals[end]
		else
			reference = func(required)
		end
			
	return collect(norm(s .- reference) for s in signals)
end

