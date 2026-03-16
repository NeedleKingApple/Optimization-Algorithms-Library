function out = AR1DA_simpson(fun, gradfun, x0, R, L, opts)
% AR1DA_simpson
% Use inexact f,g computed by simpson_nd_adaptive to minimize f(x)=∫ fun(t,x) dt
%
% fun(t,x)       : integrand for objective
% gradfun(t,x,k) : integrand for k-th partial derivative
%
% Required opts fields (AR1DA params)
%   opts.itermax
%   opts.sigma0
%   opts.sigma_min
%   opts.epsilon
%   opts.alpha
%   opts.eta1, opts.eta2
%   opts.gamma1, opts.gamma2, opts.gamma3
%
% Required Simpson fields
%   opts.simpson_f : options for function values (abs_tol will be overwritten by accu each iter)
%   opts.simpson_g : options for gradient components (rel_tol will be overwritten by omega each iter)

x = x0;
n = length(x);

% -----------------------------
% AR1DA parameters (same as your script)
% -----------------------------
itermax   = opts.itermax;

sigma     = opts.sigma0;
sigma_min = opts.sigma_min;

epsilon   = opts.epsilon;
alpha     = opts.alpha;
eta1      = opts.eta1;
eta2      = opts.eta2;
gamma1    = opts.gamma1;
gamma2    = opts.gamma2;
gamma3    = opts.gamma3;

% komega = 0.5 * alpha * eta1;
komega = 1e-2;
omega  = min(komega, 1/sigma);

like = zeros(itermax,1);

% For reporting
f = NaN; g = NaN; gnorm = NaN;

for k = 1:itermax

    % -----------------------------
    % Inexact gradient with relative accuracy omega
    % (your Simpson_partial_rel(..., omega, ...) behavior)
    % -----------------------------
    simg = opts.simpson_g;
    simg.rel_tol = omega;

    g = zeros(n,1);
    for j = 1:n
        g(j) = simpson_nd_adaptive(@(t) gradfun(t,x,j), R, L, simg);
    end

    gnorm = norm(g);
    like(k) = gnorm;

    % stopping rule (same as your script)
    if gnorm < epsilon/(1+omega)
        iternumber = k;
        break;
    end

    % -----------------------------
    % Trial step and ratio
    % -----------------------------
    s  = -g / sigma;
    DT = (g.'*g) / sigma;

    accu = max(1e-8, omega * DT);     % same as your script

    % Inexact function values with absolute accuracy accu
    % (your Simpson_f_abs(..., accu, ...) behavior)
    simf = opts.simpson_f;
    simf.abs_tol = accu;

    f  = simpson_nd_adaptive(@(t) fun(t,x),    R, L, simf);
    fs = simpson_nd_adaptive(@(t) fun(t,x+s),  R, L, simf);

    ro = (f - fs) / DT;

    % accept step
    if ro > eta1
        x = x + s;
        % you can carry f = fs, but next iteration will recompute anyway
    end

    % update sigma (same as your script)
    if ro > eta2
        sigma = max(sigma_min, gamma1*sigma);
    elseif ro > eta1
        sigma = gamma2*sigma;
    else
        sigma = gamma3*sigma;
    end

    % update omega
    omega = min(komega, 1/sigma);

    iternumber = k;
end

if ~exist('iternumber','var')
    iternumber = itermax;
end

% Final report values (optional recompute with final omega / simpson settings)
simg = opts.simpson_g;
simg.rel_tol = omega;
g = zeros(n,1);
for j = 1:n
    g(j) = simpson_nd_adaptive(@(t) gradfun(t,x,j), R, L, simg);
end
gnorm = norm(g);

simf = opts.simpson_f;
if isnan(f)
    f = simpson_nd_adaptive(@(t) fun(t,x), R, L, simf);
end

out.x     = x;
out.f     = f;
out.g     = g;
out.gnorm = gnorm;
out.iter  = iternumber;
out.like  = like(1:iternumber);

out.sigma = sigma;
out.omega = omega;
end
