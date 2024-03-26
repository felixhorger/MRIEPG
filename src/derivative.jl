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
	relaxation::ConstantRelaxation, # TODO: how would this work with multiple systems? Or varying TR
	num_systems::Integer,
	num_parameters::Integer, # number of signals that will be simulated is num_systems * (1 + 2 * num_parameters)
	rf_matrices::AbstractArray{<: Complex, 3},
	upper::Integer,
	kmax::Integer,
	mode::Union{Val{:minimal}, Val{:full}, Val{:full_in}, Val{:full_out}},
	source_state::Matrix{<: Complex},
	target_state::Matrix{<: Complex},
	recording::Union{Matrix{<: Complex}, Array{<: Complex, 3}, Nothing}
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

	# the first axis of source_state should be ordered as
	#=	[
			signal,
			<place holder p₁ system 1>, <place holder p₂ system 1> ...,
			∂sₜ₋₁/∂p₁ system 1, ∂sₜ₋₁/∂p₂ system 1, ...
		]
	=#

	# TODO: split simulate!() function into rf(), relaxation already in function, and spoiler shift
	# Then, use that in here to make more efficient
	# TODO: make apply_relaxation(target, source,...) which should be able to handle the case target==source

	num_placeholders = num_parameters * num_systems
	num_signals = 1 + 2 * num_placeholders

	# Copy sₜ₋₁(p) into place holders
	# TODO: turbo copy possible?
	for s = 1:3
		for k = 1:upper
			i = (k-1) * num_signals + 1
			j = i + num_placeholders
			@views source_state[(i+1):j, s] .= source_state[i, s]
			# i is the first index for that state (the actual signal),
			# and i+1 then is the first place holder
		end
	end

	# Apply matrices (A(p) and ∂A(p)/∂p) and relaxation r(p), 0, and ∂r(p)/∂p
	simulate!(
		t, trel,
		relaxation, num_systems,
		rf_matrices,
		upper, kmax, mode,
		source_state, target_state,
		recording
	)

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

