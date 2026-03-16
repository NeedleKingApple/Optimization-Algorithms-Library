%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% A clean benchmark script (single file) for 3 solvers:
%   - IGD   : Inexact Gradient Descent
%   - INCG  : Inexact NCG (specific beta formula)
%   - AR1DA : Adaptive Regularization with inexact evaluations
%
% Supported test functions (with per-problem default initial points):
%   1) EXTROSEN   : Extended Rosenbrock (even n)
%   2) GENROSEN   : Generalized Rosenbrock
%   3) FLETCHCR   : Fletcher function (CUTE)
%   4) PERTQUAD   : Perturbed Quadratic
%   5) BDQRTIC    : BDQRTIC (CUTE) (n>=5)
%   6) DQDRTIC    : DQDRTIC (CUTE) (n>=3)
%   7) ARWHEAD    : ARWHEAD (CUTE) (n>=2)
%   8) ENGVAL1    : ENGVAL1 (CUTE) (n>=2)
%   9) RAYDAN2    : Raydan 2
%
% Oracle model:
%   f_tilde(x) = f(x) + fun_e(k)
%   g_tilde(x) = grad_eval(fun,x,h_k) + grad_e(k)   (componentwise FD + noise)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; 
% clc;

% =======================
% Select problem
% =======================
prob.name = 'EXTROSEN';   % choose from list above
prob.n    = 100;        % dimension

x0_user = [];           % set [] to use default; or set a vector to override

% =======================
% Common settings
% =======================
tol      = 1e-6;        % termination for ||g_tilde|| (or epsilon criterion in AR1DA)
iter_max = 1e4;

% Inexactness models (customize as you like)
fun_ee  = @(k) 0;           % function noise 我们这里不引入函数值的噪声
grad_e  = @(k) 0;           % additive gradient noise (vector-level scale) 也没有梯度的额外噪声

% Finite difference step (can be schedule)
fd_h = @(k) (1e-6/k);            % central-difference step size used in grad_eval

% Choose gradient source
use_analytic_grad = false;   % if true, use analytic grad when available

% =======================
% Solver parameters
% =======================
% IGD
igd.tau    = 2e-2; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
igd.lambda = 0.5;
igd.delta  = 1e-3;
igd.tol_ls = @(k) 0;  % inexact Armijo tolerance term

% INCG
incg.tau    = 1e-3; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
incg.lambda = 0.5;
incg.delta  = 1e-3;
incg.tol_ls = @(k) 0;
incg.u1     = 1.3;
incg.u2     = 0.2;

% AR1DA
ar.sigma0    = 1;
ar.sigma_min = 1e-3;
ar.alpha     = 1e-3;      % alpha in (0,1)
ar.eta1      = 0.1;
ar.eta2      = 0.8;
ar.gamma1    = 0.05;
ar.gamma2    = 1.2;
ar.gamma3    = 2.5;

% =======================
% Build problem
% =======================
[fun, grad_analytic, x0_default] = get_problem(prob.name, prob.n);

if isempty(x0_user)
    x0 = x0_default;
else
    x0 = x0_user;
end

% fprintf('Problem: %s, n = %d\n', upper(prob.name), prob.n);
% fprintf('||x0|| = %.3e\n', norm(x0));

% Choose gradient oracle
if use_analytic_grad && ~isempty(grad_analytic)
    grad_oracle = @(x,k) grad_analytic(x) + grad_e(k)*ones(prob.n,1);
else
    grad_oracle = @(x,k) grad_eval_fd(fun, x, fd_h(k)) + grad_e(k)*ones(prob.n,1);
end

% Function oracle
fun_oracle = @(x,k) fun(x) + fun_ee(k);

% =======================
% Run 3 solvers
% =======================
fprintf('\nRunning IGD...\n');
tic;
[outIGD] = IGD(fun_oracle, grad_oracle, x0, tol, iter_max, igd);
tIGD = toc;

fprintf('Running AR1DA...\n');
tic;
[outAR] = AR1DA(fun_oracle, grad_oracle, x0, tol, iter_max, ar);
tAR = toc;

fprintf('Running INCG...\n');
tic;
[outINCG] = INCG(fun_oracle, grad_oracle, x0, tol, iter_max, incg);
tINCG = toc;

% =======================
% Report
% =======================
fprintf('\n==================== RESULTS ====================\n');
fprintf('current dimension: n = %d\n', prob.n);
fprintf('IGD  : iter = %d, f = %.6e, ||g|| = %.3e, time = %.4fs\n', outIGD.iter, outIGD.f, outIGD.gnorm, tIGD);
fprintf('AR1DA: iter = %d, f = %.6e, ||g|| = %.3e, time = %.4fs\n', outAR.iter,  outAR.f,  outAR.gnorm,  tAR);
fprintf('INCG : iter = %d, f = %.6e, ||g|| = %.3e, time = %.4fs\n', outINCG.iter,outINCG.f,outINCG.gnorm,tINCG);
fprintf('=================================================\n');

% % =======================
% % Plot histories
% % =======================
% figure; hold on; box on; grid on;
% semilogy(outIGD.g_hist,'LineWidth',1.5);
% semilogy(outAR.g_hist,'LineWidth',1.5);
% semilogy(outINCG.g_hist,'LineWidth',1.5);
% xlabel('Iteration'); ylabel('||g\_tilde||');
% legend({'IGD','AR1DA','INCG'},'Location','northeast');
% title(sprintf('%s (n=%d): gradient norm history', upper(prob.name), prob.n));
% 
% figure; hold on; box on; grid on;
% plot(outIGD.f_hist,'LineWidth',1.5);
% plot(outAR.f_hist,'LineWidth',1.5);
% plot(outINCG.f_hist,'LineWidth',1.5);
% xlabel('Iteration'); ylabel('f\_tilde(x)');
% legend({'IGD','AR1DA','INCG'},'Location','northeast');
% title(sprintf('%s (n=%d): function value history', upper(prob.name), prob.n));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SOLVERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = IGD(fun_oracle, grad_oracle, x0, tol, iter_max, prm)
% Inexact Gradient Descent with inexact Armijo line search.

x = x0;

tau    = prm.tau;
lambda = prm.lambda;
delta  = prm.delta;
tol_ls = prm.tol_ls;

f_hist = zeros(iter_max,1);
g_hist = zeros(iter_max,1);

for k = 1:iter_max
    f = fun_oracle(x,k);
    g = grad_oracle(x,k);
    gnorm = norm(g);

    f_hist(k) = f;
    g_hist(k) = gnorm;

    if gnorm < tol
        break;
    end

    alpha = tau;
    f0 = f;

    while true
        fnew = fun_oracle(x - alpha*g, k);
        if fnew < f0 - delta*alpha*(gnorm^2) + tol_ls(k)
            break;
        end
        alpha = alpha*lambda;
        if alpha < 1e-16, break; end
    end

    x = x - alpha*g;
end

out.x = x;
out.f = fun_oracle(x,k);
out.gnorm = norm(grad_oracle(x,k));
out.iter = k;
out.f_hist = f_hist(1:k);
out.g_hist = g_hist(1:k);
end


function out = INCG(fun_oracle, grad_oracle, x0, tol, iter_max, prm)
% Your INCG variant:
%   d_k = -g_k + beta_k * (d_{k-1} - proj_{g_k} d_{k-1})
%   beta_k = (u1*||g_k||^2) / (u2*||d_{k-1}||*||g_k-g_{k-1}|| + ||g_{k-1}||^2)
% Armijo-type inexact line search along d_k.

x = x0;

tau    = prm.tau;
lambda = prm.lambda;
delta  = prm.delta;
tol_ls = prm.tol_ls;
u1     = prm.u1;
u2     = prm.u2;

f_hist = zeros(iter_max,1);
g_hist = zeros(iter_max,1);

g_prev = [];
d_prev = [];

for k = 1:iter_max

    f = fun_oracle(x,k);
    g = grad_oracle(x,k);
    gnorm = norm(g);

    f_hist(k) = f;
    g_hist(k) = gnorm;

    if gnorm < tol
        break;
    end

    if k == 1
        d = -g;
    else
        beta_num = u1 * (g'*g);
        beta_den = u2 * norm(d_prev) * norm(g - g_prev) + (g_prev'*g_prev);
        beta = beta_num / max(beta_den, 1e-16);

        % perpendicular component
        d_prev_perp = d_prev - g * ( (g'*d_prev) / max(g'*g, 1e-16) );
        d = -g + beta * d_prev_perp;
    end

    alpha = tau;
    f0 = f;

    while true
        fnew = fun_oracle(x + alpha*d, k);
        if fnew < f0 + delta*alpha*(g'*d) + tol_ls(k)
            break;
        end
        alpha = alpha*lambda;
        if alpha < 1e-16, break; end
    end

    x = x + alpha*d;

    g_prev = g;
    d_prev = d;
end

out.x = x;
out.f = fun_oracle(x,k);
out.gnorm = norm(grad_oracle(x,k));
out.iter = k;
out.f_hist = f_hist(1:k);
out.g_hist = g_hist(1:k);
end


function out = AR1DA(fun_oracle, grad_oracle, x0, tol, iter_max, prm)
% AR1DA-style adaptive regularization with inexact evaluations (spirit of Alg 6.1):
%   s_k = -g_k / sigma_k
%   DT_k = ||g_k||^2 / sigma_k
%   rho_k = (f(x_k) - f(x_k + s_k))/DT_k
%   sigma update with gamma1/gamma2/gamma3
% Termination: ||g_k|| < tol

x = x0;

sigma     = prm.sigma0;
sigma_min = prm.sigma_min;
alpha     = prm.alpha;
eta1      = prm.eta1;
eta2      = prm.eta2;
gamma1    = prm.gamma1;
gamma2    = prm.gamma2;
gamma3    = prm.gamma3;

% (kept for alignment; not explicitly used with this oracle model)
komega = 0.5*alpha*eta1; %#ok<NASGU>

f_hist = zeros(iter_max,1);
g_hist = zeros(iter_max,1);

for k = 1:iter_max

    f = fun_oracle(x,k);
    g = grad_oracle(x,k);
    gnorm = norm(g);

    f_hist(k) = f;
    g_hist(k) = gnorm;

    if gnorm < tol
        break;
    end

    s  = -g / sigma;
    DT = (g'*g) / sigma;

    fs = fun_oracle(x + s, k);
    rho = (f - fs) / max(DT, 1e-16);

    if rho > eta1
        x = x + s;   % accept
        f = fs;
    end

    if rho > eta2
        sigma = max(sigma_min, gamma1*sigma);
    elseif rho > eta1
        sigma = gamma2*sigma;
    else
        sigma = gamma3*sigma;
    end
end

out.x = x;
out.f = fun_oracle(x,k);
out.gnorm = norm(grad_oracle(x,k));
out.iter = k;
out.f_hist = f_hist(1:k);
out.g_hist = g_hist(1:k);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PROBLEM FACTORY (ONLY the requested 9 functions)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fun, grad, x0] = get_problem(name, n)
% Return fun(x), optional analytic grad(x) (empty if not provided),
% and the per-problem default initial point x0 aligned to your original script.

name = upper(strtrim(name));
grad = [];     % optional analytic gradient
x0   = zeros(n,1);

switch name

    % ============================================================
    % Extended Rosenbrock (even n)
    % f = sum_{i=1}^{n/2} [ 100(x_{2i} - x_{2i-1}^2)^2 + (x_{2i-1}-1)^2 ]
    % x0: odd=-1.2, even=1
    % ============================================================
    case {'EXTROSEN','EXTENDED_ROSENBROCK'}
        if mod(n,2) ~= 0
            error('%s requires even n.', name);
        end
        fun = @(x) sum( 100*(x(2:2:end) - x(1:2:end).^2).^2 + (x(1:2:end) - 1).^2 );
        x0 = zeros(n,1);
        x0(1:2:end) = -1.2;
        x0(2:2:end) = 1;

    % ============================================================
    % Generalized Rosenbrock
    % f = sum_{i=1}^{n-1} [ 100(x_{i+1}-x_i^2)^2 + (x_i-1)^2 ]
    % x0: pair-style (-1.2,1,...) in your script
    % ============================================================
    case {'GENROSEN','GENERALIZED_ROSENBROCK','ROSEN'}
        fun = @(x) sum( 100*(x(2:end) - x(1:end-1).^2).^2 + (x(1:end-1) - 1).^2 );
        x0 = -1.2*ones(n,1);
        x0(2:2:end) = 1;

    % ============================================================
    % FLETCHCR (CUTE)
    % f = sum 100*(x_{i+1}-x_i+1-x_i^2)^2
    % x0: zeros(n,1)
    % ============================================================
    case {'FLETCHCR'}
        fun = @(x) sum( 100*(x(2:end) - x(1:end-1) + 1 - x(1:end-1).^2).^2 );
        x0  = zeros(n,1);

    % ============================================================
    % Perturbed Quadratic
    % f = sum i*x_i^2 + (sum x_i)^2/100
    % x0: 0.5*ones
    % ============================================================
    case {'PERTQUAD','PERTURBED_QUADRATIC','PQ'}
        fun = @(x) sum( (1:n)'.*(x.^2) ) + (sum(x).^2)/100;
        x0  = 0.5*ones(n,1);

    % ============================================================
    % BDQRTIC (CUTE)  (n>=5)
    % f = sum [(-4 x_i + 3)^2 + (x_i^2 + 2x_{i+1}^2 + 3x_{i+2}^2 + 4x_{i+3}^2 + 5x_{i+4}^2)]
    % x0: ones
    % ============================================================
    case {'BDQRTIC'}
        if n < 5, error('BDQRTIC requires n>=5.'); end
        fun = @(x) sum( (-4*x(1:end-4)+3).^2 + ...
            (x(1:end-4).^2 + 2*x(2:end-3).^2 + 3*x(3:end-2).^2 + 4*x(4:end-1).^2 + 5*x(5:end).^2) );
        x0 = ones(n,1);

    % ============================================================
    % DQDRTIC (CUTE) (n>=3)
    % f = sum [ x_{i}^2 + 100 x_{i+1}^2 + 100 x_{i+2}^2 ], i=1..n-2
    % x0: 3*ones
    % ============================================================
    case {'DQDRTIC'}
        if n < 3, error('DQDRTIC requires n>=3.'); end
        fun = @(x) sum( x(1:end-2).^2 + 100*x(2:end-1).^2 + 100*x(3:end).^2 );
        x0  = 3*ones(n,1);

        % optional analytic gradient
        grad = @(x) dqdrtic_grad(x);

    % ============================================================
    % ARWHEAD (CUTE) (n>=2)
    % f = sum_{i=1}^{n-1} [ -4 x_i + 3 + (x_i^2 + x_n^2)^2 ]
    % x0: ones
    % ============================================================
    case {'ARWHEAD'}
        if n < 2, error('ARWHEAD requires n>=2.'); end
        fun = @(x) sum( (-4*x(1:end-1)+3) + (x(1:end-1).^2 + x(end).^2).^2 );
        x0  = ones(n,1);

    % ============================================================
    % ENGVAL1 (CUTE) (n>=2)
    % f = sum_{i=1}^{n-1} (x_i^2 + x_{i+1}^2)^2 + sum_{i=1}^{n-1} (-4 x_i + 3)
    % x0: 2*ones (as in your script)
    % ============================================================
    case {'ENGVAL1'}
        if n < 2, error('ENGVAL1 requires n>=2.'); end
        fun = @(x) sum( (x(1:end-1).^2 + x(2:end).^2).^2 ) + sum( -4*x(1:end-1) + 3 );
        x0  = 2*ones(n,1);

    % ============================================================
    % Raydan 2
    % f = sum(exp(x) - x)
    % x0: ones
    % ============================================================
    case {'RAYDAN2'}
        fun = @(x) sum(exp(x) - x);
        x0  = ones(n,1);

    otherwise
        error('Unknown problem: %s', name);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ORACLE UTILITIES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function g = grad_eval_fd(fun, x, h)
% Central difference gradient
n = length(x);
g = zeros(n,1);
for i = 1:n
    xp = x; xm = x;
    xp(i) = xp(i) + h;
    xm(i) = xm(i) - h;
    g(i) = (fun(xp) - fun(xm)) / (2*h);
end
end

function g = dqdrtic_grad(x)
% Analytic gradient for DQDRTIC:
% f = sum_{i=1}^{n-2} [ x_i^2 + 100 x_{i+1}^2 + 100 x_{i+2}^2 ].
% So each component accumulates from up to 3 neighboring terms.

n = length(x);
g = zeros(n,1);

% Contributions:
% x(i) appears in:
%   - term i      with coeff 1   (as x_i^2) for i <= n-2
%   - term i-1    with coeff 100 (as x_{i}^2 = x_{(i-1)+1}^2) for i-1 >= 1 => i >= 2 and (i-1)<=n-2 => i<=n-1
%   - term i-2    with coeff 100 (as x_{i}^2 = x_{(i-2)+2}^2) for i-2 >= 1 => i >= 3 and (i-2)<=n-2 => i<=n

for i = 1:n
    coeff = 0;
    if i <= n-2
        coeff = coeff + 1;
    end
    if (i >= 2) && (i <= n-1)
        coeff = coeff + 100;
    end
    if i >= 3
        coeff = coeff + 100;
    end
    g(i) = 2*coeff*x(i);
end
end
