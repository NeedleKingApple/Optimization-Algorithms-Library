%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Supported 9 problems (with per-problem default initial points):
%   1) EXTROSEN : Extended Rosenbrock (even n)
%   2) GENROSEN : Generalized Rosenbrock
%   3) FLETCHCR : Fletcher function (CUTE)
%   4) PERTQUAD : Perturbed Quadratic
%   5) BDQRTIC  : BDQRTIC (CUTE) (n>=5)
%   6) DQDRTIC  : DQDRTIC (CUTE) (n>=3)
%   7) ARWHEAD  : ARWHEAD (CUTE) (n>=2)
%   8) ENGVAL1  : ENGVAL1 (CUTE) (n>=2)
%   9) RAYDAN2  : Raydan 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;
% clc;

% =======================
% Select problem
% =======================
prob.name = 'DQDRTIC';   % {'EXTROSEN','GENROSEN','FLETCHCR','PERTQUAD','BDQRTIC','DQDRTIC','ARWHEAD','ENGVAL1','RAYDAN2'}
prob.n    = 2000;

% Use per-problem default initial point unless user overrides:
x0_user = [];             % [] -> use default; otherwise provide vector

% =======================
% INCG-F parameters (match your Simpson test script style)
% =======================
opts.tau     = 1e-3;
opts.lambda  = 0.5;
opts.sigma   = 1e-3;
opts.itermax = 1e4;

% truncation/scaling params (ensure nu*bar_gamma >= 1)
opts.nu        = 1e3;
opts.bar_gamma = 1e3;

% stopping rule (consistent pattern)
opts.stop_tol_g = 1e-6;

% =======================
% Gradient-source settings
% =======================
use_analytic_grad = false;   % true -> use analytic grad if implemented; otherwise fallback FD
Ntest = 1;

% =======================
% Build problem
% =======================
[fun_exact, grad_analytic, x0_default] = get_problem_9(upper(prob.name), prob.n);

if isempty(x0_user)
    x0 = x0_default;
else
    x0 = x0_user;
end

% =======================
% beta sweep (FIXED)
% =======================
% beta_list = {'HS','PRP','FR','DY','HZ'};   % <-- must be cell array

beta_list = {'FR'};
for i = 1:numel(beta_list)

    opts.beta_type = beta_list{i};        % <-- cell indexing

    t_sum = 0;
    it_sum = 0;
    out_last = [];

    for rep = 1:Ntest

        % define oracles (ONLY source differs; algorithm identical)
        fun_oracle  = @(x,k) fun_exact(x); % no function noise

        if use_analytic_grad && ~isempty(grad_analytic)
            grad_oracle = @(x,k) grad_analytic(x);
        else
            % central FD with schedule h_k = 1e-6 / max(1,k)
            grad_oracle = @(x,k) grad_fd_central(fun_exact, x, 1e-6 / max(1,k));
        end

        tic;
        out = INCG_F_oracle(fun_oracle, grad_oracle, x0, opts);
        t_sum = t_sum + toc;
        it_sum = it_sum + out.iter;
        out_last = out;
    end

    avg_time = t_sum / Ntest;
    avg_iter = it_sum / Ntest;
    final_gn = out_last.gnorm;

    % ===== NEW: final function value =====
    final_f  = out_last.f;      % out_last must contain field .f
    avg_f    = final_f;         % if Ntest>1, change to average over reps (see note below)

    % ---- FIXED fprintf format (with f) ----
    fprintf(['beta_type = %s | dimension = %d | avg_time = %.4f s | avg_iter = %.2f', ...
             ' | f = %.6e | last_gnorm = %.3e\n'], ...
        opts.beta_type, prob.n, avg_time, avg_iter, avg_f, final_gn);


end

% Result = table(omega_list(:), avg_iter(:), avg_time(:), final_gn(:), ...
%     'VariableNames', {'omega','avg_iter','avg_time','last_gnorm'});
% disp(Result);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INCG-F ORACLE SOLVER (IDENTICAL to your INCG_F_simpson except oracle calls)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = INCG_F_oracle(fun_oracle, grad_oracle, x0, opts)
% fun_oracle(x,k)  : returns f(x) (k is passed for schedule consistency)
% grad_oracle(x,k) : returns g(x) (possibly approximate)
%
% Algorithmic structure is the SAME as INCG_F_simpson.

x = x0;
n = length(x);

tau     = opts.tau;
lambda  = opts.lambda;
sigma   = opts.sigma;
itermax = opts.itermax;

nu        = opts.nu;
bar_gamma = opts.bar_gamma;

stop_tol_g = opts.stop_tol_g;

beta_type = 'HS'; % 这是默认选项
if isfield(opts,'beta_type'), beta_type = upper(opts.beta_type); end

like = zeros(itermax,1);

% initial f(x0)
f = fun_oracle(x,0);

% initial gradient gtilde_0
gtilde = grad_oracle(x,0); % k = 0

gnorm = norm(gtilde);
like(1) = gnorm;

% initial direction
d_prev = -gtilde;
gtilde_prev = gtilde;

if gnorm < stop_tol_g
    out.x = x;
    out.f = f;
    out.gnorm = gnorm;
    out.iter = 0;
    out.like = like(1);
    return
end

for k = 1:itermax

    like(k) = gnorm;

    if gnorm < stop_tol_g
        break
    end

    % -------------------------
    % build d_k from d_{k-1}, gtilde_k
    % -------------------------
    if k == 1
        d_k = d_prev;
    else
        % compute d_{k-1}^hbar = (I - gg^T/||g||^2) d_{k-1}
        if norm(gtilde) == 0
            d_h = zeros(n,1);
        else
            d_h = d_prev - gtilde * ( (gtilde.'*d_prev) / (gtilde.'*gtilde) );
        end

        % ytilde = gtilde_k - gtilde_{k-1}
        ytilde = gtilde - gtilde_prev;

        % compute beta_k according to chosen formula (using inexact info)
        beta = compute_beta_5(beta_type, gtilde, gtilde_prev, ytilde, d_prev);

        % truncation bound bar_beta_k
        if norm(d_h) == 0
            bar_beta = inf;
        else
            tmp = nu^2 * bar_gamma^2 - 1;
            if tmp < 0
                error('Need nu^2*bar_gamma^2 - 1 >= 0. Please ensure nu*bar_gamma >= 1.');
            end
            bar_beta = sqrt(tmp) * norm(gtilde) / norm(d_h);
        end

        % truncated beta
        beta_tilde = min(max(beta, -bar_beta), bar_beta);

        % direction
        d_k = -gtilde + beta_tilde * d_h;
    end

    % -------------------------
    % scaling: p_k = d_k / gamma_k
    % -------------------------
    if norm(gtilde) == 0
        gamma_k = 1;
    else
        gamma_k = max(1, norm(d_k)/(nu*norm(gtilde)));
    end
    gamma_k = min(gamma_k, bar_gamma);

    p_k = d_k / gamma_k;

    % -------------------------
    % Armijo-type inexact line search (fixed initial step tau)
    % -------------------------
    alpha = tau;
    f0 = f;

    % tol schedule identical to your Simpson version
    tol_f = 0;
    if isfield(opts,'tol_f0') && ~isempty(opts.tol_f0)
        tol_f = opts.tol_f0 / max(1,k)^2;
    end
    tol_ls = 5 * tol_f;

    while true
        x_trial = x + alpha * p_k;
        fnew = fun_oracle(x_trial,k);

        if fnew <= f0 + sigma*alpha*(gtilde.'*p_k) + tol_ls
            break
        end

        alpha = lambda * alpha;
        if alpha < 1e-16
            break
        end
    end

    % update
    x = x + alpha * p_k;
    f = fnew;

    % update gradient
    gtilde_prev = gtilde;
    gtilde = grad_oracle(x,k);
    gnorm = norm(gtilde);

    % shift direction
    d_prev = d_k;

end

out.x = x;
out.f = f;
out.gnorm = gnorm;
out.iter = k;
out.like = like(1:k);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% beta computation helper (5 choices): HS/PRP/FR/DY/HZ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function beta = compute_beta_5(beta_type, gk, gkm1, yk, dkm1)
switch upper(beta_type)

    case 'HS'
        denom = dkm1.'*yk;
        if abs(denom) < 1e-16
            beta = 0;
        else
            beta = (gk.'*yk) / denom;
        end

    case 'PRP'
        denom = gkm1.'*gkm1;
        if abs(denom) < 1e-16
            beta = 0;
        else
            beta = (gk.'*yk) / denom;
        end

    case 'FR'
        denom = gkm1.'*gkm1;
        if abs(denom) < 1e-16
            beta = 0;
        else
            beta = (gk.'*gk) / denom;
        end

    case 'DY'
        denom = dkm1.'*yk;
        if abs(denom) < 1e-16
            beta = 0;
        else
            beta = (gk.'*gk) / denom;
        end

    case 'HZ'
        % Hager–Zhang (HZ):
        % beta = ( (y - 2 d * ||y||^2/(d^T y))^T gk ) / (d^T y)
        denom = dkm1.' * yk;
        if abs(denom) < 1e-16
            beta = 0;
        else
            y_norm2 = yk.' * yk;
            y_hat   = yk - 2 * dkm1 * (y_norm2 / denom);
            beta    = (y_hat.' * gk) / denom;
        end

    otherwise
        error('Unknown beta_type: %s', beta_type);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PROBLEM FACTORY (ONLY the requested 9 functions) + default x0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fun, grad, x0] = get_problem_9(name, n)
name = upper(strtrim(name));
grad = [];
x0   = zeros(n,1);

switch name

    % Extended Rosenbrock (even n)
    case {'EXTROSEN','EXTENDED_ROSENBROCK'}
        if mod(n,2) ~= 0
            error('%s requires even n.', name);
        end
        fun = @(x) sum(100*(x(2:2:end) - x(1:2:end).^2).^2 + (x(1:2:end) - 1).^2);
        x0 = zeros(n,1);
        x0(1:2:end) = -1.2;
        x0(2:2:end) = 1;

        % Generalized Rosenbrock
    case {'GENROSEN','GENERALIZED_ROSENBROCK','ROSEN'}
        fun = @(x) sum(100*(x(2:end) - x(1:end-1).^2).^2 + (x(1:end-1) - 1).^2);
        x0 = -1.2*ones(n,1);
        x0(2:2:end) = 1;

        % Fletcher function (CUTE): FLETCHCR
    case {'FLETCHCR','FLETCHER'}
        fun = @(x) sum(100*(x(2:end) - x(1:end-1) + 1 - x(1:end-1).^2).^2);
        x0  = zeros(n,1);

        % Perturbed Quadratic
    case {'PERTQUAD','PERTURBED_QUADRATIC','PQ'}
        fun = @(x) sum((1:n)'.*(x.^2)) + (sum(x).^2)/100;
        x0  = 0.5*ones(n,1);

        % BDQRTIC (CUTE) (n>=5)
    case {'BDQRTIC'}
        if n < 5, error('BDQRTIC requires n>=5.'); end
        fun = @(x) sum( (-4*x(1:end-4)+3).^2 + ...
            (x(1:end-4).^2 + 2*x(2:end-3).^2 + 3*x(3:end-2).^2 + 4*x(4:end-1).^2 + 5*x(5:end).^2) );
        x0 = ones(n,1);

        % DQDRTIC (CUTE) (n>=3)
    case {'DQDRTIC'}
        if n < 3, error('DQDRTIC requires n>=3.'); end
        fun = @(x) sum(x(1:end-2).^2 + 100*x(2:end-1).^2 + 100*x(3:end).^2);
        x0  = 3*ones(n,1);
        % ARWHEAD (CUTE) (n>=2)
    case {'ARWHEAD'}
        if n < 2, error('ARWHEAD requires n>=2.'); end
        fun = @(x) sum((-4*x(1:end-1)+3) + (x(1:end-1).^2 + x(end).^2).^2);
        x0  = ones(n,1);

        % ENGVAL1 (CUTE) (n>=2)
    case {'ENGVAL1'}
        if n < 2, error('ENGVAL1 requires n>=2.'); end
        fun = @(x) sum((x(1:end-1).^2 + x(2:end).^2).^2) + sum(-4*x(1:end-1) + 3);
        x0  = 2*ones(n,1);

        % Raydan 2
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

function g = grad_fd_central(fun, x, h)
% Central difference gradient with step size h
n = length(x);
g = zeros(n,1);
for i = 1:n
    xp = x; xm = x;
    xp(i) = xp(i) + h;
    xm(i) = xm(i) - h;
    g(i) = (fun(xp) - fun(xm)) / (2*h);
end
end