
struct PartialGrid{T <: Real}
	q1::Vector{T} # q for quantity, e.g. it can be R1 or E1
	q2::Vector{T}
	Δ::Vector{Int64}
	function PartialGrid(q1::AbstractVector{T}, q2::AbstractVector{T}, Δ::AbstractVector{<: Integer}) where T <: Real
		@assert length(q1) == maximum(Δ)
		@assert length(q2) == length(Δ)
		return new{T}(q1, q2, Δ)
	end
	function PartialGrid(q1::AbstractVector{T}, q2::AbstractVector{T}) where T <: Real
		# Only use combinations where q2 > q1
		# q1,2 have to be sorted
		Δ = Vector{Int64}(undef, length(q2))
		Δ .= 0
		local j = 1
		for i = 1:length(q2)
			for outer j = j:length(q1)
				q1[j] > q2[i] && break
			end
			Δ[i] = j
		end
		return new{T}(q1, q2, Δ)
	end
end 

import Base.length
@inline function Base.length(g::PartialGrid)
	sum(g.Δ)
end

