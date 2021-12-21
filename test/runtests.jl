
using Test
using MAT
import PyPlot as plt

import MRIEPG

function main()
	# Compare EPG-X simulation (matlab and cpp) to the julia version

	# Read in matlab's results
	file = matopen("simulation.mat")
	kmax = Int(read(file, "kmax"))
	matlab_epgs = read(file, "matlab_epgs")
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
	close(file)

	# Simulate in julia
	R = MRIEPG.PartialGrid(1 ./ T1, 1 ./ T2)
	initial_state = repeat(initial_state, R.num_systems)
	#println(MRIEPG.diffusion_factor.(MRIEPG.diffusion_b_values(G, τ, kmax)[1], D)')
	#println(MRIEPG.diffusion_factor.(MRIEPG.diffusion_b_values(G, τ, kmax)[2], D)')
	@time epgs, _ = MRIEPG.simulate(Val(:full), kmax, α, ϕ, TR, G, τ, D, R, initial_state, Val(:all))

	# Compare
	signals = transpose(@view epgs[1:R.num_systems, 2, :])

	plt.figure()
	plt.imshow(abs.(epgs[:, 2, :] .- matlab_epgs[:, 2, :]))
	
	plt.figure()
	plt.plot(abs.(signals[:, 1]); label="Julia")
	plt.plot(abs.(matlab_epgs[1, 2, :]); label="Matlab")
	plt.legend()
	plt.show()
end

main()

