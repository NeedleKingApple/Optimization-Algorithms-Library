% clc; 
clear;

n = 4;
R = zeros(n,1);
L = ones(n,1);
x0 = ones(n,1);

% -----------------------------
% Simpson options
% -----------------------------
opts.simpson_f.abs_tol = 1e-6;     % will be overwritten by accu each iter
opts.simpson_f.rel_tol = 0;
opts.simpson_f.max_refine = 50;
opts.simpson_f.level_factor = 5;

opts.simpson_g.abs_tol = 1e-12;
opts.simpson_g.rel_tol = 0.1;      % will be overwritten by omega each iter
opts.simpson_g.max_refine = 50;
opts.simpson_g.level_factor = 5;

% -----------------------------
% AR1DA parameters (same spirit as your script)
% -----------------------------
opts.itermax   = 200;
opts.sigma0    = 1;
opts.sigma_min = 1e-3;

opts.epsilon = 1e-6;
opts.alpha   = 1e-3;
opts.eta1    = 0.1;
opts.eta2    = 0.8;
opts.gamma1  = 0.05;
opts.gamma2  = 1.2;
opts.gamma3  = 2.5;


tic
out = AR1DA_simpson(@integrand, @integrand_grad, x0, R, L, opts);
toc

fprintf('iters = %d\n', out.iter);
fprintf('||g||  = %.3e\n', out.gnorm);
fprintf('f(x)   = %.16e\n', out.f);
fprintf('x      = [%s]\n', num2str(out.x.'));

figure;
semilogy(out.like,'LineWidth',1.5);
xlabel('Iteration'); ylabel('||g_k||'); grid on;
