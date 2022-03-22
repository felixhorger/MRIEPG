
using Test
using MAT
import PyPlot as plt

import MRIEPG

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
	R = MRIEPG.XLargerYs.XLargerY(1 ./ reverse(T1), 1 ./ reverse(T2))
	
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
					Val(:full), kmax,
					α, ϕ, TR,
					G, τ, D,
					(R.q1[m], R.q2[n]),
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
					Val(:full), kmax,
					α, ϕ, fill(TR, length(α)),
					G, τ, D,
					(R.q1[m], R.q2[n]),
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
			Val(:full), kmax,
			α, ϕ, TR,
			G, τ, D,
			R,
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
			Val(:full), kmax,
			α, ϕ, fill(TR, length(α)),
			G, τ, D,
			R,
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
end

main()

