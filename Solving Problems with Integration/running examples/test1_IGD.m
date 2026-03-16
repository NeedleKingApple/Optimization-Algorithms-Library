clc; clear;

% =======================
% Problem setup
% =======================
n = 3;
R = zeros(n,1);
L = ones(n,1);
x0 = ones(n,1);

% =======================
% IGD parameters (fixed)
% =======================
opts.tau     = 3.5;
opts.lambda  = 0.5;
opts.delta   = 1e-3;
opts.itermax = 1e3;

% =======================
% Simpson parameters for f (fixed)
% =======================
opts.simpson_f.abs_tol      = 0.5;
opts.simpson_f.rel_tol      = 0;
opts.simpson_f.max_refine   = 20;
opts.simpson_f.level_factor = 5;

% =======================
% Simpson parameters for g (rel_tol will vary)
% =======================
opts.simpson_g.abs_tol      = 1e-10;
opts.simpson_g.max_refine   = 20;
opts.simpson_g.level_factor = 5;

% test grid for rel_tol
% rel_list = [0.3, 0.2, 0.1, 0.01, 0.001];

rel_list = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3];
Ntest = 1;

% results
avg_time = zeros(length(rel_list),1);
avg_iter = zeros(length(rel_list),1);

% =======================
% Run experiments
% =======================
for i = 1:length(rel_list)
    opts.simpson_g.rel_tol = rel_list(i);

    % IGD stopping threshold depends on rel_tol
    opts.stop_tol_g = 1e-6 / (opts.simpson_g.rel_tol + 1);

    iters = zeros(Ntest,1);

    tic;
    for k = 1:Ntest
        out = IGD_simpson(@integrand, @integrand_grad, x0, R, L, opts);
        iters(k) = out.iter;
    end
    total_time = toc;

    avg_time(i) = total_time / Ntest;
    avg_iter(i) = mean(iters);

    fprintf('rel_tol = %.1e | avg_time = %.4f s | avg_iter = %.2f\n', ...
        rel_list(i), avg_time(i), avg_iter(i));
end

% % =======================
% % Show as a table
% % =======================
% T = table(rel_list(:), avg_time, avg_iter, ...
%     'VariableNames', {'simpson_g_rel_tol','avg_time_sec','avg_iter'});
% disp(T);
% 
% % =======================
% % (Optional) plots
% =======================
figure;
semilogx(rel_list, avg_time, '-o','LineWidth',1.5);
xlabel('opts.simpson\_g.rel\_tol');
ylabel('Average time per run (sec)');
grid on;

figure;
semilogx(rel_list, avg_iter, '-o','LineWidth',1.5);
xlabel('opts.simpson\_g.rel\_tol');
ylabel('Average iterations');
grid on;
