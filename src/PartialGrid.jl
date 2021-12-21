
struct PartialGrid{T <: Real}
	q1::Vector{T} # q for quantity, e.g. it can be R1 or E1
	q2::Vector{T}
	Δ::Vector{Int64}
	num_systems::Int64 # Rename this some time to something more abstract

	function PartialGrid(q1::AbstractVector{T}, q2::AbstractVector{T}, Δ::AbstractVector{<: Integer}) where T <: Real
		@assert all(Δ .> 0)
		@assert length(q1) >= maximum(Δ)
		@assert length(q2) == length(Δ)
		return new{T}(q1, q2, Δ, sum(Δ))
	end
	function PartialGrid(q1::AbstractVector{T}, q2::AbstractVector{T}) where T <: Real
		# Only use combinations where q2 > q1
		sort!(q1)
		sort!(q2)
		Δ = Vector{Int64}(undef, length(q2))
		local i
		local j = 1
		@inbounds for outer i = 1:length(q2)
			for outer j = j:length(q1)
				q1[j] > q2[i] && break
			end
			if q1[j] > q2[i]
				j -= 1
			else
				break
			end
			Δ[i] = j
		end
		# Fill remaining elements up with maximum possible Δ
		Δ[i:end] .= length(q1)
		return new{T}(q1, q2, Δ, sum(Δ))
	end
end 



macro iterate_partial_grid(partial_grid, loop_body_a, loop_body_b)
	# In each iteration, the respective array elements are in i:j
	# loop_body_a: do everything independent of q1, since q2 is constant over i:j
	# To get the respective q2 = partial_grid.q2[n]
	# variable names: 
	#	- index n in the array partial_grid.q2
	# loop_body_b: q1 varies (variable name q1)
	# To get the respective q1 = $partial_grid.q1[m]
	# variable names:
	#	- index m in the array partial_grid.q1
	#	- index l corresponding index in i:j
	esc(quote
		local i = 1
		for (n, δ) = enumerate($partial_grid.Δ)
			j = i + δ - 1
			$loop_body_a
			for (m, l) = enumerate(i:j)
				$loop_body_b
			end
			i += δ
		end
	end)
end



function get_combinations(g::PartialGrid{T})::Matrix{T} where T <: Real
	qs = Matrix{T}(undef, 2, g.num_systems)
	@iterate_partial_grid(
		g,
		nothing,
		begin
			qs[1, l] = g.q1[m]
			qs[2, l] = g.q2[n]
		end
	)
	return qs
end

