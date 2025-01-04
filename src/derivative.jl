function simulate_derivative(
	TR::Real,
	R::NTuple{2, <: Real},
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	kmax::Integer
)

here 	compute_relaxation(
		TR::Real,
		R::NTuple{2, <: Real},
		G::AbstractVector{<: Real},
		τ::AbstractVector{<: Real},
		D::Real,
		kmax::Integer
	)


	simulate!(
		t, trel,
		relaxation, num_systems,
		rf_matrices,
		upper, kmax, Val(:minimal),
		source_state, target_state,
		recording
	)

end



function simulate_derivative!(
	t::Integer,
	trel::Integer,
	relaxation::Union{ConstantRelaxation}, # TODO: how would this work with multiple systems? Or varying TR
	relaxation_derivative::Union{ConstantRelaxation}, # TODO add other relaxation structs
	num_systems::Integer,
	num_parameters::Integer, # number of signals that will be simulated is num_systems * (1 + 2 * num_parameters)
	rf_matrices::AbstractArray{<: Complex, 3},
	upper::Integer,
	kmax::Integer,
	mode::Union{Val{:minimal}, Val{:full}, Val{:full_in}, Val{:full_out}},
	source_state::Matrix{<: Complex},
	target_state::Matrix{<: Complex},
	source_state_derivative::Matrix{<: Complex},
	target_state_derivative::Matrix{<: Complex},
	recording::Union{Matrix{<: Complex}, Array{<: Complex, 3}, Nothing},
	recording_derivative::Union{Matrix{<: Complex}, Array{<: Complex, 3}, Nothing}
)
	#=
		sₜ(p) = A(p) ⋅ sₜ₋₁(p) + r(p)
		∂sₜ(p)/∂p = [ ∂A(p)/∂p ⋅ sₜ₋₁(p) ] + [ A(p) ⋅ ∂sₜ₋₁(p)/∂p + ∂r(p)/∂p ]

		Note that signal is recorded directly after the pulse, i.e. the derivative w.r.t. T2, 
		due to TE is not considered.

		TODO:
		\partial_{T_2} \tilde{s}_t(p) = \partial_{T_2} \, R \cdot s_{t-1}(p) \cdot e^{-T_E/T_2} =\\
		R \, \cdot \left( e^{-T_E/T_2} \cdot \partial_{T_2} \, s_{t-1}(p) \ + \  s_{t-1}(p) \cdot \partial_{T_2} \, e^{-T_E/T_2} \right) =\\
		e^{-T_E/T_2} \cdot R \, \cdot \left( \partial_{T_2} \, s_{t-1}(p) \ + \  s_{t-1}(p) \cdot \, T_E/T_2^2 \right)

		Note that the factor $e^{-T_E/T_2}$ needs to be included in all other derivatives as well,
		otherwise wrong scaling
	=#

	TODO: state = [signal, dp1, dp2, ..., k=1:3]
	apply rf (single call)
	record all (single call)
	apply derivative relaxation to state[1:n_params+1:end, :] to get (∂A/∂p s), can I reuse state[1:n, :] bc non-consecutive? Otherwise need separate array
	apply longitudinal recovery to signal in state
	apply derivative longitudinal recovery to derivative in state or to (∂A/∂p s), for this apply the function to state[2:numparams+1:end, :] or could just write it myself, copying the contents of apply_longitudinal_recovery!(). This might also be better since for T1/T2 something different needs to be done.
	add (∂A/∂p s) to the derivative in state
	apply gradient shift to state (single call)



	num_signals = (1 + num_parameters) * num_systems
	upper_systems = num_signals * upper

	# 1) Apply pulse matrices and record signal
	apply_rf_pulse_matrices!.((
		(rf_matrices, source_state,            target_state,            trel, upper_systems),
		(rf_matrices, source_state_derivative, target_state_derivative, trel, upper_systems)
	))
	record_state!.((
		(recording,            target_state,            t, num_signals),
		(recording_derivative, target_state_derivative, t, num_signals)
	))

	# 2) Apply relaxation
	# ∂A(p)/∂p: target_state -> source state
	apply_relaxation!(source_state, target_state, derivative_relaxation, upper, kmax, trel)

	apply_longitudinal_recovery!(target_state, relaxation, trel)


	gradient_shift!(target_state, num_systems, upper_systems)






	# Apply matrices (A(p) and ) and relaxation r(p), 0, and ∂r(p)/∂p

	# Apply rule to calculate derivative for target state ...
	for s = 1:3
		for k = 1:upper
			i = (k-1) * num_signals + 1
			j = i + num_placeholders
			m = j + num_placeholders
			@views target_state[(j+1):m, s] .+= target_state[(i+1):j, s]
			# i is the first index for that state (the actual signal),
			# and i+1 then is the first place holder
			# j+1 is the first index of where the derivatives are stored
		end
	end

	return 
end

