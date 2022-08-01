
using Test
using MAT
import PyPlot as plt

import MRIEPG

function test_time()
	t0 = 1
	timepoints = 121
	G = [0.0, 3.0, 42.0, 0.0]
	τ = [9.0, 3.5, 0.5, 7.0]
	D = 2e-12
	R1 = 1/1000
	R2 = 1/100
	R1s = 1 ./ [100, 200, 500, 1000, 2000]
	R2s = 1 ./ [10, 50, 100, 1000]
	R1s, R2s = reverse!.((R1s, R2s))
	R = MRIEPG.XLargerYs.XLargerY(R2s, R1s)
	TR = Vector{Float64}(undef, timepoints)
	TR[1:3] .= 30.0
	TR[4:end-1] .= 10.0
	TR[end] = 500.0
	kmax = 50
	relaxation, num_systems = MRIEPG.compute_relaxation(TR, R, G, τ, D, kmax)
	rf_matrices = zeros(ComplexF64, 3, 3, timepoints)
	memory = MRIEPG.allocate_memory(Val(:minimal), timepoints, num_systems, kmax, nothing, Val(:signal))
	MRIEPG.simulate!(t0, timepoints, rf_matrices, relaxation, num_systems, kmax, Val(:minimal), memory)
	@time MRIEPG.simulate!(t0, timepoints, rf_matrices, relaxation, num_systems, kmax, Val(:minimal), memory)
	@time MRIEPG.simulate!(t0, timepoints, rf_matrices, relaxation, num_systems, kmax, Val(:minimal), memory)
	# Twice because compilation
end

function main()
	# Compare EPG-X simulation (matlab and cpp) to the julia version

	# Read in matlab's results
	file = matopen("simulation.mat")
	kmax = Int(read(file, "kmax"))
	epgs_matlab = read(file, "epgs_matlab")
	α = dropdims(read(file, "alpha"); dims=2)
	ϕ = dropdims(read(file, "phi"); dims=2)
	TR = read(file, "TR")
	T1 = read(file, "T1")
	if T1 isa Number
		T1 = [T1]
	else
		T1 = dropdims(T1; dims=1)
	end
	T2 = read(file, "T2")
	if T2 isa Number
		T2 = [T2]
	else
		T2 = dropdims(T2; dims=1)
	end
	G = dropdims(read(file, "G"); dims=1)
	τ = dropdims(read(file, "tau"); dims=1)
	D = 1e-3 * read(file, "D")
	initial_state = read(file, "initial_state")
	D_longitudinal_matlab = read(file, "D_longitudinal")
	D_transverse_matlab = read(file, "D_transverse")
	close(file)


	# Simulate in julia
	# Diffusion factors
	let
		D_longitudinal, D_transverse = MRIEPG.diffusion_b_values(G, τ, kmax)
		D_longitudinal = MRIEPG.diffusion_factor.(D_longitudinal, D)
		D_transverse = MRIEPG.diffusion_factor.(D_transverse, D)
		@test isapprox(D_longitudinal_matlab, D_longitudinal, rtol=1e-4)
		@test isapprox(D_transverse_matlab, D_transverse, rtol=1e-4)
	end

	# Set up relaxivities
	R = MRIEPG.XLargerYs.XLargerY(1 ./ reverse(T2), 1 ./ reverse(T1))
	
	# ConstantRelaxation
	# TODO: Check minimal, full_in, full_out.
	# Maybe compare original EPG with my kmax modification, then also simulate only minimal
	let
		epgs = Array{ComplexF64}(undef, kmax+1, 3, length(α), R.N)
		MRIEPG.XLargerYs.@iterate(
			R,
			# Outer loop
			nothing,
			# Inner loop
			begin
				epgs[:, :, :, l], _ = MRIEPG.simulate(
					α, ϕ, TR,
					(R.y[m], R.x[n]),
					G, τ, D,
					kmax, Val(:full),
					initial_state,
					Val(:all)
				)
			end
		)
		#plt.figure()
		#plt.imshow(abs.(epgs[:, 2, :, 1]))
		#plt.imshow(imag.(epgs[:, 2, :, 1]))
		#plt.figure()
		#plt.imshow(imag.(epgs[:, 1, :, 1]))
		#plt.imshow(abs.(epgs_matlab[:, 2, :, 1]))
		#plt.figure()
		#plt.imshow(abs.(epgs[:, 2, :, 1] - epgs_matlab[:, 2, :, 1]))
		#plt.figure()
		#plt.imshow(angle.(epgs[:, 2, :, 1] - epgs_matlab[:, 2, :, 1]))
		#plt.show()
		@test isapprox(epgs, epgs_matlab, rtol=1e-4)
	end

	# MultiTRRelaxation
	# TODO: Check minimal, full_in, full_out.
	let
		epgs = Array{ComplexF64}(undef, kmax+1, 3, length(α), R.N)
		MRIEPG.XLargerYs.@iterate(
			R,
			nothing,
			begin
				epgs[:, :, :, l], _ = MRIEPG.simulate(
					α, ϕ, fill(TR, length(α)),
					(R.y[m], R.x[n]),
					G, τ, D,
					kmax, Val(:full),
					initial_state,
					Val(:all)
				)
			end
		)
		@test isapprox(epgs, epgs_matlab, rtol=1e-4)
	end


	# MultiSystemRelaxation
	# TODO: Check minimal, full_in, full_out.
	let
		epgs, _ = MRIEPG.simulate(
			α, ϕ, TR,
			R,
			G, τ, D,
			kmax, Val(:full),
			repeat(initial_state, inner=(R.N, 1)),
			Val(:all)
		)
		epgs = reshape(epgs, (R.N, kmax+1, 3, length(α)))
		epgs = permutedims(epgs, (2, 3, 4, 1))
		@test isapprox(epgs, epgs_matlab, rtol=1e-4)
	end

	# MultiSystemMultiTRRelaxation
	# TODO: Check minimal, full_in, full_out.
	let
		epgs, _ = MRIEPG.simulate(
			α, ϕ, fill(TR, length(α)),
			R,
			G, τ, D,
			kmax, Val(:full),
			repeat(initial_state, inner=(R.N, 1)),
			Val(:all)
		)
		epgs = reshape(epgs, (R.N, kmax+1, 3, length(α)))
		epgs = permutedims(epgs, (2, 3, 4, 1))
		@test isapprox(epgs, epgs_matlab, rtol=1e-4)
	end

	#=
	initial_state = repeat(initial_state, R.N)
	@time epgs, _ = MRIEPG.simulate(Val(:full), kmax, α, ϕ, TR, G, τ, D, R, initial_state, Val(:all))

	# Compare
	@views signals = transpose(epgs[1:R.N, 2, :])

	plt.figure()
	plt.imshow(abs.(epgs[:, 2, :] .- matlab_epgs[:, 2, :]))
	#@test isapprox(epgs, epgs_matlab; rtol=0.05)
	
	plt.figure()
	plt.plot(abs.(signals[:, 1]); label="Julia")
	plt.plot(abs.(epgs_matlab[1, 2, :]); label="Matlab")
	plt.legend()
	plt.show()
	=#

	# Driven equilibrium
	let
		cycles = 8
		G = [0.0, 3.0, 42.0, 0.0]
		τ = [9.0, 3.5, 0.5, 7.0]
		D = 2e-12
		R1 = 1/1000
		R2 = 1/100
		R1s = 1 ./ [100, 200, 500, 1000, 2000]
		R2s = 1 ./ [10, 50, 100, 1000]
		R1s, R2s = reverse!.((R1s, R2s))
		R = MRIEPG.XLargerYs.XLargerY(R2s, R1s)
		kmax = 50
		α = [[π, 0, 0]; deg2rad.(1:60); deg2rad.(59:-1:2)]
		ϕ = zeros(length(α))
		TR = Vector{Float64}(undef, length(α))
		TR[1:3] .= 30.0
		TR[4:end-1] .= 10.0
		TR[end] = 500.0
		local s = Array{ComplexF64, 2}(undef, length(R), cycles * length(TR))
		i = 1
		for r2 in R2s
			for r1 in R1s
				r1 ≥ r2 && continue
				tmp = MRIEPG.simulate(
					repeat(α, cycles),
					repeat(ϕ, cycles),
					repeat(TR, cycles),
					(r1, r2),
					G, τ, D,
					kmax,
					Val(:minimal)
				)
				s[i, :] = tmp
				i += 1
			end
		end
		s2 = MRIEPG.driven_equilibrium(cycles, α, ϕ, TR, R, G, τ, D, kmax, Val(:signal))
		@assert all(v -> isapprox(v[1], v[2]; atol=2^(floor(log2(abs(v[1])))-52)), zip(s, s2))
		# Note on choice of atol: one bit flipped because of x86 register spill
	end

end

main()

