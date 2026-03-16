clear; close all;
% clc;

NTest = 1;
% =======================
% Select problem
% =======================
prob.name = 'RAYDAN2';
prob.n    = 1000;

x0_user = [];   % [] -> use default

% =======================
% Common settings
% =======================
tol      = 1e-6;
iter_max = 1e4;

fun_ee  = @(k) 0;        % no function noise
grad_e  = @(k) 0;        % no additive gradient noise
fd_h    = @(k) 1e-5; %  (1e-6/max(1,k));   % FD stepsize schedule

use_analytic_grad = false;
subNewton.m        = 20;                  % L-BFGS memory
subNewton.rmax     = 20;            % max subspace dimension
subNewton.restart  = 30;                  % restart frequency
subNewton.tau      = 1;
subNewton.lambda   = 0.5;
subNewton.delta    = 1e-4;
subNewton.ls_maxit = 20;
subNewton.nu_thr   = 1e-8;
subNewton.tol = 1e-6;
subNewton.iter_max = 1e4;





% =======================
% Build problem
% =======================
[fun, grad_analytic, x0_default] = get_problem(prob.name, prob.n);

if isempty(x0_user)
    x0 = x0_default;
else
    x0 = x0_user;
end

% Choose gradient oracle
if use_analytic_grad && ~isempty(grad_analytic)
    grad_oracle = @(x,k) grad_analytic(x) + grad_e(k)*ones(prob.n,1);
else
    grad_oracle = @(x,k) grad_eval_fd(fun, x, fd_h(k)) + grad_e(k)*ones(prob.n,1);
end

fun_oracle = @(x,k) fun(x) + fun_ee(k);





fprintf('Running SubNewton...\n');
tic;
for i = 1:NTest
outSubNewton = SubNewton(fun_oracle, grad_oracle, x0, subNewton);
end
tSubNewton = toc;

fprintf('SubNewton: iter = %d, f = %.6e, ||g|| = %.3e, time = %.4fs\n', outSubNewton.iter, outSubNewton.f, outSubNewton.gnorm, tSubNewton/NTest);



% SubNewton: subspace L-BFGS with inexact line search (noise-compatible)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = SubNewton(fun_oracle, grad_oracle, x0, opts)
% SubNewton:
%   - Subspace built from recent gradients (restartable)
%   - L-BFGS update performed in the subspace
%   - Inexact Armijo-type backtracking (same oracle model)

x = x0(:);
n = length(x);

% -------- parameters (LOCAL, do not touch benchmark params) ----------
m        = opts.m;                  % L-BFGS memory
rmax     = min(opts.rmax,n);            % max subspace dimension
restart  = opts.restart;                  % restart frequency
tau      = opts.tau;
lambda   = opts.lambda;
delta    = opts.delta;
ls_maxit = opts.ls_maxit;
nu_thr   = opts.nu_thr;
tol = opts.tol;
iter_max = opts.iter_max;

% -------- storage ----------
f_hist = zeros(iter_max,1);
g_hist = zeros(iter_max,1);

% subspace + L-BFGS memory
Z = [];
S = [];
Y = [];
RHO = [];
current_size = 0;
ptr = 1;

label = 0;
x_prev = [];
g_prev = [];
ghat_prev = [];

for k = 1:iter_max
    f = fun_oracle(x,k);
    g = grad_oracle(x,k);
    gnorm = norm(g);

    f_hist(k) = f;
    g_hist(k) = gnorm;

    if gnorm < tol
        break;
    end

    % -------- build / restart subspace ----------
    if isempty(Z) || mod(label, restart) == 0
        if gnorm == 0
            Z = eye(n,1);
        else
            Z = g / gnorm;
        end

        % reset L-BFGS memory
        S = zeros(1,m);
        Y = zeros(1,m);
        RHO = zeros(1,m);
        current_size = 0;
        ptr = 1;

        ghat_prev = [];
        x_prev = x;
    else
        u = Z.'*g;
        zz = g - Z*u;
        rou = norm(zz);

        if rou > nu_thr*gnorm && size(Z,2) < rmax
            zz = zz / max(rou,1e-16);
            Z = [Z, zz];

            % expand memory matrices
            S = [S; zeros(1,m)];
            Y = [Y; zeros(1,m)];
        end
    end

    r = size(Z,2);
    ghat = Z.'*g;

    if isempty(ghat_prev)
        % do nothing
    else
        rp = length(ghat_prev);
        if rp < r
            ghat_prev = [ghat_prev; zeros(r-rp,1)];
        elseif rp > r
            ghat_prev = ghat_prev(1:r);
        end
    end

    % -------- L-BFGS update in subspace ----------
    if ~isempty(ghat_prev)
        s_hat = Z.'*(x - x_prev);
        y_hat = ghat - ghat_prev;

        if s_hat'*y_hat > 1e-12*norm(s_hat)*norm(y_hat)
            rho = 1/(y_hat'*s_hat);

            if current_size == m
                S(:,ptr) = s_hat;
                Y(:,ptr) = y_hat;
                RHO(ptr) = rho;
                ptr = mod(ptr,m)+1;
            else
                current_size = current_size + 1;
                S(:,current_size) = s_hat;
                Y(:,current_size) = y_hat;
                RHO(current_size) = rho;
            end
        end
    end

    % -------- compute direction ----------
    d_hat = lbfgs_subspace_dir(ghat, S, Y, RHO, current_size, ptr, m);
    d = Z*d_hat;

    if g'*d >= 0
        d = -g;
    end

    % -------- inexact backtracking ----------
    alpha = tau;
    for ls = 1:ls_maxit
        fnew = fun_oracle(x + alpha*d, k);
        if fnew <= f + delta*alpha*(g'*d)
            break;
        end
        alpha = lambda*alpha;
    end

    % -------- accept ----------
    x_prev = x;
    g_prev = g;
    ghat_prev = ghat;

    x = x + alpha*d;
    label = label + 1;
end

out.x = x;
out.f = fun_oracle(x,k);
out.gnorm = norm(grad_oracle(x,k));
% out.gnorm = gnorm;
out.iter = k;
out.f_hist = f_hist(1:k);
out.g_hist = g_hist(1:k);
end


function d = lbfgs_subspace_dir(g, S, Y, RHO, current_size, ptr, m)
r = length(g);

if current_size == 0
    d = -g;
    return;
end

q = g;
alpha = zeros(current_size,1);

if current_size == m
    idxs = mod((ptr-2)+(1:current_size),m)+1;
else
    idxs = 1:current_size;
end

for i = current_size:-1:1
    idx = idxs(i);
    alpha(i) = RHO(idx)*(S(:,idx)'*q);
    q = q - alpha(i)*Y(:,idx);
end

if current_size == m
    last = mod(ptr-1,m)+1;
else
    last = current_size;
end

ys = Y(:,last)'*S(:,last);
yy = Y(:,last)'*Y(:,last);
gamma = ys/max(yy,1e-16);

rvec = gamma*q;

for i = 1:current_size
    idx = idxs(i);
    beta = RHO(idx)*(Y(:,idx)'*rvec);
    rvec = rvec + S(:,idx)*(alpha(i)-beta);
end

d = -rvec;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TRUST-REGION SUBPROBLEM SOLVER (Steihaug-CG) in subspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [s, val] = tr_steihaug_subspace(g, B, delta, tol, itmax_factor)
r = length(g);
s = zeros(r,1);

rvec = g;
p = -rvec;
rr = rvec.'*rvec;

if sqrt(rr) < tol
    val = 0;
    return;
end

itmax = max(5, itmax_factor*r);

for k = 1:itmax
    Bp = B*p;
    pBp = p.'*Bp;

    if pBp <= 0
        tau = boundary_tau(s, p, delta);
        s = s + tau*p;
        break;
    end

    alpha = rr / pBp;
    s_next = s + alpha*p;

    if norm(s_next) >= delta
        tau = boundary_tau(s, p, delta);
        s = s + tau*p;
        break;
    end

    s = s_next;
    rvec = rvec + alpha*Bp;
    rr_new = rvec.'*rvec;

    if sqrt(rr_new) < tol
        break;
    end

    beta = rr_new / rr;
    p = -rvec + beta*p;
    rr = rr_new;
end

val = g.'*s + 0.5*s.'*B*s;
end

function tau = boundary_tau(s, p, delta)
a = p.'*p;
b = 2*(s.'*p);
c = s.'*s - delta^2;
disc = max(b^2 - 4*a*c, 0);
tau = (-b + sqrt(disc)) / (2*a);
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PROBLEM FACTORY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fun, grad, x0] = get_problem(name, n)
name = upper(strtrim(name));
grad = [];
x0   = zeros(n,1);

switch name
    case {'EXTROSEN','EXTENDED_ROSENBROCK'}
        if mod(n,2) ~= 0, error('%s requires even n.', name); end
        fun = @(x) sum( 100*(x(2:2:end) - x(1:2:end).^2).^2 + (x(1:2:end) - 1).^2 );
        x0 = zeros(n,1); x0(1:2:end) = -1.2; x0(2:2:end) = 1;

    case {'GENROSEN','GENERALIZED_ROSENBROCK','ROSEN'}
        fun = @(x) sum( 100*(x(2:end) - x(1:end-1).^2).^2 + (x(1:end-1) - 1).^2 );
        x0 = -1.2*ones(n,1); x0(2:2:end) = 1;

    case {'FLETCHCR'}
        fun = @(x) sum( 100*(x(2:end) - x(1:end-1) + 1 - x(1:end-1).^2).^2 );
        x0  = zeros(n,1);

    case {'PERTQUAD','PERTURBED_QUADRATIC','PQ'}
        fun = @(x) sum( (1:n)'.*(x.^2) ) + (sum(x).^2)/100;
        x0  = 0.5*ones(n,1);

    case {'BDQRTIC'}
        if n < 5, error('BDQRTIC requires n>=5.'); end
        fun = @(x) sum( (-4*x(1:end-4)+3).^2 + ...
            (x(1:end-4).^2 + 2*x(2:end-3).^2 + 3*x(3:end-2).^2 + 4*x(4:end-1).^2 + 5*x(5:end).^2) );
        x0 = ones(n,1);

    case {'DQDRTIC'}
        if n < 3, error('DQDRTIC requires n>=3.'); end
        fun = @(x) sum( x(1:end-2).^2 + 100*x(2:end-1).^2 + 100*x(3:end).^2 );
        x0  = 3*ones(n,1);
        grad = @(x) dqdrtic_grad(x);

    case {'ARWHEAD'}
        if n < 2, error('ARWHEAD requires n>=2.'); end
        fun = @(x) sum( (-4*x(1:end-1)+3) + (x(1:end-1).^2 + x(end).^2).^2 );
        x0  = ones(n,1);

    case {'ENGVAL1'}
        if n < 2, error('ENGVAL1 requires n>=2.'); end
        fun = @(x) sum( (x(1:end-1).^2 + x(2:end).^2).^2 ) + sum( -4*x(1:end-1) + 3 );
        x0  = 2*ones(n,1);

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
n = length(x);
g = zeros(n,1);
for i = 1:n
    coeff = 0;
    if i <= n-2, coeff = coeff + 1; end
    if (i >= 2) && (i <= n-1), coeff = coeff + 100; end
    if i >= 3, coeff = coeff + 100; end
    g(i) = 2*coeff*x(i);
end
end
