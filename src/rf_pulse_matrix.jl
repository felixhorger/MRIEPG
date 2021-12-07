
function rf_pulse_matrix(
	α::Real,
	ϕ::Real;
	out=Matrix{ComplexF64}(undef, 3, 3)
)::Matrix{ComplexF64}
	# This is the transposed version of the one from Weigel's paper
	phase_factor = exp(im * ϕ)
	phase_factor_squared = phase_factor^2
	im_phase_factor = im * phase_factor
	@inbounds begin
		out[1, 1] = cos(0.5α)^2
		out[2, 1] = sin(0.5α)^2 * phase_factor_squared
		out[3, 1] = -sin(α) * im_phase_factor
		out[1, 2] = conj(out[2, 1])
		out[2, 2] = out[1, 1]
		out[3, 2] = conj(out[3, 1])
		out[1, 3] = -0.5 * out[3, 2]
		out[2, 3] = conj(out[1, 3])
		out[3, 3] = cos(α)
	end
	return out
end

