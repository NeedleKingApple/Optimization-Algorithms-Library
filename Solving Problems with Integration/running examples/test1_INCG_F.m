% clc; clear;

% -----------------------------
% problem setting
% -----------------------------
n = 3;
R = zeros(n,1);
L = ones(n,1);

x0 = ones(n,1);

% -----------------------------
% INCG parameters
% -----------------------------
opts.tau     = 2.7;
opts.lambda  = 0.5;
opts.sigma   = 1e-3;     % Armijo parameter (typical: 1e-4 ~ 1e-2)
opts.itermax = 1e3;

% INCG truncation/scaling params (ensure nu*bar_gamma >= 1)
opts.nu        = 1e4;
opts.bar_gamma = 1e4;

% beta rule: choose among 'HS','PRP','FR','DY','HZ'
opts.beta_type = 'HZ';

% optional: decay f-oracle abs_tol like tol_f0/k^2
opts.tol_f0 = 0.5;

% -----------------------------
% Simpson options (function vs gradient)
% -----------------------------
opts.simpson_f.abs_tol      = 0.5;
opts.simpson_f.rel_tol      = 0;
opts.simpson_f.max_refine   = 20;
opts.simpson_f.level_factor = 5;

% baseline for gradient oracle (we will sweep rel_tol)
opts.simpson_g.abs_tol      = 1e-10;
opts.simpson_g.max_refine   = 20;
opts.simpson_g.level_factor = 5;

% sweep list
% rel_list = [0.3, 0.2, 0.1, 0.01, 0.001];
rel_list = [0.2];

Ntest = 1;

avg_time = zeros(size(rel_list));
avg_iter = zeros(size(rel_list));
final_gn = zeros(size(rel_list));

for i = 1:numel(rel_list)
    opts.simpson_g.rel_tol = rel_list(i);

    % stop criterion consistent with your previous choice:
    opts.stop_tol_g = 1e-6 / (opts.simpson_g.rel_tol + 1);

    t_sum = 0;
    it_sum = 0;

    out_last = [];

    for k = 1:Ntest
        tic;
        out = INCG_F_simpson(@integrand, @integrand_grad, x0, R, L, opts);
        t_sum = t_sum + toc;
        it_sum = it_sum + out.iter;
        out_last = out;
    end

    avg_time(i) = t_sum / Ntest;
    avg_iter(i) = it_sum / Ntest;
    final_gn(i) = out_last.gnorm;

    fprintf('rel_tol = %.1e | avg_time = %.4f s | avg_iter = %.2f | last_gnorm = %.3e\n', ...
        rel_list(i), avg_time(i), avg_iter(i), final_gn(i));
end

% show as a table
Result = table(rel_list(:), avg_iter(:), avg_time(:), final_gn(:), ...
    'VariableNames', {'rel_tol_g','avg_iter','avg_time','last_gnorm'});
disp(Result);
