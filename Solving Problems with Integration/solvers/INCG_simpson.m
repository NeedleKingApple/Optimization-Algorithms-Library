
function out = INCG_simpson(fun, gradfun, x0, R, L, opts)
% INCG_simpson: Inexact nonlinear CG with Simpson-based inexact f/g
%
% fun(t,x)        : integrand of objective
% gradfun(t,x,j)  : integrand of j-th gradient component
% x0              : initial point (n×1)
% R,L             : bounds (n×1)
% opts fields (recommended):
%   opts.tau, opts.lambda, opts.delta, opts.itermax
%   opts.tol_f0               (for tol_f schedule)
%   opts.stop_tol_g           (stop when ||g|| < stop_tol_g)
%   opts.u1, opts.u2          (beta parameters)
%   opts.simpson_f, opts.simpson_g  (structs passed into simpson_nd_adaptive)

x = x0;
n = length(x);

% parameters
tau     = opts.tau;
lambda  = opts.lambda;
delta   = opts.delta;
itermax = opts.itermax;
tol_f0  = opts.simpson_f.abs_tol;

if ~isfield(opts,'tol_f0'),      opts.tol_f0 = 0.5; end
if ~isfield(opts,'stop_tol_g'),  opts.stop_tol_g = 1e-6; end
if ~isfield(opts,'u1'),          opts.u1 = 1.2; end
if ~isfield(opts,'u2'),          opts.u2 = 0.3; end

u1 = opts.u1;
u2 = opts.u2;

like = zeros(itermax,1);
% initial f
f = simpson_nd_adaptive(@(t) fun(t,x), R, L, opts.simpson_f);

% initial g
g = zeros(n,1);
for j = 1:n
    g(j) = simpson_nd_adaptive(@(t) gradfun(t,x,j), R, L, opts.simpson_g);
end

% initial direction
d = -g;

for k = 1:itermax

    gnorm = norm(g);
    like(k) = gnorm;
    if gnorm < opts.stop_tol_g
        break
    end

    % tolerance schedule for inexact line search (match your code)
    tol_f  = tol_f0 / (k^2);
    opts.simpson_f.abs_tol = tol_f;
    tol_ls = 5 * tol_f;

    % Armijo-type inexact line search along d
    alpha = tau;
    f0 = f;

    while true
        fnew = simpson_nd_adaptive(@(t) fun(t, x + alpha*d), R, L, opts.simpson_f);
        if fnew <= f0 + delta * alpha * (g.'*d) + tol_ls
            break
        end
        alpha = lambda * alpha;
        % (optional safeguard)
        if alpha < 1e-16
            break
        end
    end

    % update x, f
    x = x + alpha*d;
    f = fnew;

    % compute new gradient
    g_old = g;
    g = zeros(n,1);
    for j = 1:n
        g(j) = simpson_nd_adaptive(@(t) gradfun(t,x,j), R, L, opts.simpson_g);
    end

    % compute beta (your formula)
    d_old = d;
    beta_num = u1 * (g.'*g);
    % like(k) = g.'*g /( norm(g)*(g.'*g_old)./norm(g_old));
    beta_den = u2 * norm(d_old) * norm(g - g_old) + (g_old.'*g_old);
    if beta_den <= 0
        beta = 0;
    else
        beta = beta_num / beta_den;
    end

    % orthogonalize d_old w.r.t. current g (avoid division by zero)
    gg = g.'*g;
    if gg > 0
        d_old_perp = d_old - g * ((g.'*d_old) / gg);
    else
        d_old_perp = d_old;
    end

    % new direction
    d = -g + beta * d_old_perp;
end
opts.simpson_f.abs_tol = tol_f0;
out.x = x;
out.f = f;
out.g = g;
out.gnorm = norm(g);
out.iter = k;
out.like = like(1:k);
end
