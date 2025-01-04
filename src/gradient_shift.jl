@inbounds @views function gradient_shift!(state::Matrix{<: Complex}, num_systems::Integer, upper_systems::Integer)
	# TODO rename uppersystems?

	total_num_states = size(state, 1)

	# F^+(k) goes to F^+(k+1), only for k > 0
	# The first i is the first index of the k-state which is newly populated by the shift
	#println("+") # Excuse the shotgun debug comment clutter
	for i = min(upper_systems, total_num_states-num_systems) + 1 : -num_systems : num_systems+1
		state[i:i+num_systems-1, 1] .= state[i-num_systems : i-1, 1]
		#println("$(i-num_systems : i-1) -> $(i:i+num_systems-1)")
	end

	# F^+(k = 0) becomes the conjugate of F^-(k = -1)
	state[1:num_systems, 1] .= conj.(state[num_systems+1 : 2*num_systems, 2])

	# F^-(k) also goes to F^-(k+1), but k is ordered the other way around in the array
	#println("-")
	for i = 1:num_systems:(upper_systems-num_systems)
		state[i : i+num_systems-1, 2] .= state[i+num_systems : i+2*num_systems-1, 2]
		#println("$(i+num_systems : i+2*num_systems-1) -> $(i : i+num_systems-1)")
	end

	# F^-(k = -upper+1) is unpopulated
	state[upper_systems-num_systems+1 : upper_systems, 2] .= 0

	# Longitudinal states are not getting shifted

	return
end

