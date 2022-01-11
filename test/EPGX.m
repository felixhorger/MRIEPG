
function epgs_matlab = EPGX()
	% Requires: Fork of the EPG-X matlab package (https://github.com/felixhorger/EPG-X)

	timepoints = 100;
	alpha = pi/2 * ones(timepoints, 1);
	phi = 2 * pi * ones(timepoints, 1);
	TR = 20.0; % [ms]
	N = 10;
	T1 = linspace(100, 5000, N);
	T2 = linspace(10, 2500, N);

	G = [0.0, 10.0];
	tau = [15.0, TR - 15.0];
	D = 2e-9;
	diffusion.G = G;
	diffusion.tau = tau;
	diffusion.D = D;

	kmax = timepoints - 1;
	num_states = 3*(kmax+1);
	%initial_state = rand([num_states 1]) + 1i * rand([num_states 1]);
	% For simpler debugging: Thermal equilibrium state
	initial_state = zeros([num_states 1]);
	initial_state(3) = 1;

	% Get diffusion factors
	[D_longitudinal, D_transverse] = EPG_diffusion_weights(G, tau, D, 0:kmax);

	% Get how many T1,2 combinations are valid
	count = 0;
	for j = N:-1:1 % Other direction because in julia code, R1,2 is used (sorted!)
		for i = N:-1:1
			if T1(i) < T2(j)
				break
			end
			count = count + 1;
		end
	end

	epgs_matlab = zeros(kmax+1, 3, timepoints, count);
	count = 0;
	for j = N:-1:1 % Other direction because in julia code, R1,2 is used (sorted!)
		for i = N:-1:1
			if T1(i) < T2(j)
				break
			end
			count = count + 1;
			[~, ~, ~, recording, ~] = EPG_GRE(alpha, phi, TR, T1(i), T2(j), 'kmax', kmax, 'diff', diffusion, 'state', initial_state);
			epgs_matlab(:, :, :, count) = recording;
		end
	end

	[dirname, ~] = fileparts(mfilename('fullpath'));
	initial_state = reshape(initial_state, 3, kmax+1).';
	% systems and k, component, timepoints
	save(fullfile(dirname, "simulation.mat"), 'epgs_matlab', 'alpha', 'phi', 'TR', 'T1', 'T2', 'G', 'tau', 'D', 'kmax', 'initial_state', 'D_longitudinal', 'D_transverse', '-v7.3')
end

