clc; 
clear;

% ===============================
% problem dimension and bounds
% ===============================
n = 3;
R = zeros(n,1);
L = ones(n,1);

% initial point
x0 = ones(n,1);

% ===============================
% INCG algorithm parameters
% ===============================
base_opts.tau     = 4;
base_opts.lambda  = 0.5;
base_opts.delta   = 1e-3;
base_opts.itermax = 500;

base_opts.u1      = 1.2;
base_opts.u2      = 0.3;

% inexactness schedule
base_opts.tol_f0  = 0.5;

% ===============================
% Simpson options for f
% ===============================
base_opts.simpson_f.abs_tol      = 0.5;
base_opts.simpson_f.rel_tol      = 0;
base_opts.simpson_f.max_refine   = 20;
base_opts.simpson_f.level_factor = 5;

% ===============================
% Simpson options for g (abs part fixed)
% ===============================
base_opts.simpson_g.abs_tol      = 1e-10;
base_opts.simpson_g.max_refine   = 20;
base_opts.simpson_g.level_factor = 5;

% ===============================
% test settings
% ===============================
rel_tol_list = [0.3, 0.2, 0.1, 0.01, 0.001];

Ntest = 1;

fprintf('INCG tests (n = %d, Ntest = %d)\n', n, Ntest);
fprintf('-----------------------------------------------\n');

for rt = rel_tol_list

    total_time = 0;
    total_iter = 0;

    for k = 1:Ntest
        opts = base_opts;

        % set rel_tol for gradient integration
        opts.simpson_g.rel_tol = rt;

        opts.stop_tol_g = 1e-6 / (rt + 1);

        tic;
        out = INCG_simpson(@integrand, @integrand_grad, x0, R, L, opts);
        t = toc;

        total_time = total_time + t;
        total_iter = total_iter + out.iter;
    end

    avg_time = total_time / Ntest;
    avg_iter = total_iter / Ntest;

    fprintf('rel_tol = %.1e | avg_time = %.4f s | avg_iter = %.2f\n', ...
            rt, avg_time, avg_iter);
end

fprintf('-----------------------------------------------\n');
