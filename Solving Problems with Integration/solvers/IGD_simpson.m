function out = IGD_simpson(fun, gradfun, x0, R, L, opts)
% fun      : integrand F(t,x)
% gradfun  : integrand gradient component ∂F/∂x_k
% x0       : initial point
% R,L      : integration bounds
% opts     : algorithm parameters
%
% Required fields:
%   opts.simpson_f : Simpson options for function values
%   opts.simpson_g : Simpson options for gradient components

x = x0;
n = length(x);

tau     = opts.tau;
lambda  = opts.lambda;
delta   = opts.delta;
stop_tol_g = opts.stop_tol_g;
tol_f0  = opts.simpson_f.abs_tol;
itermax = opts.itermax;

like = zeros(itermax,1);

% ============================
% initial function value (use simpson_f)
% ============================
f = simpson_nd_adaptive(@(t) fun(t,x), R, L, opts.simpson_f);

for k = 1:itermax

    % tolerance schedule for inexact line search
    tol_f  = tol_f0 / k^2;
    opts.simpson_f.abs_tol = tol_f;
    tol_ls = 5 * tol_f;

    % ============================
    % compute inexact gradient (use simpson_g)
    % ============================
    g = zeros(n,1);
    for j = 1:n
        g(j) = simpson_nd_adaptive( ...
            @(t) gradfun(t,x,j), R, L, opts.simpson_g);
    end

    gnorm = norm(g);
    like(k) = gnorm;

    % stopping criterion
    if gnorm < stop_tol_g
        break
    end

    % ============================
    % Armijo-type inexact line search
    % (function values still use simpson_f)
    % ============================
    alpha = tau;
    f0 = f;

    while true
        fnew = simpson_nd_adaptive( ...
            @(t) fun(t, x - alpha*g), R, L, opts.simpson_f);

        if fnew <= f0 - delta*alpha*gnorm^2 + tol_ls
            break
        end
        alpha = lambda * alpha;
    end

    % gradient step
    x = x - alpha * g;
    f = fnew;
end
opts.simpson_f.abs_tol = tol_f0;
out.x     = x;
out.f     = f;
out.gnorm = gnorm;
out.iter  = k;
out.like  = like(1:k);
end
