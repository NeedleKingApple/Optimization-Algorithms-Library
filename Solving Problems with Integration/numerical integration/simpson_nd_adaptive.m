function [I, info] = simpson_nd_adaptive(fun, R, L, opts)
% fun  : function handle, fun(t) where t is n×1
% R,L  : bounds (n×1)
% opts : struct with fields:
%        abs_tol (default 1e-6)
%        rel_tol (default 1e-6)
%        max_refine (default 20)
%        level_factor (default 5)

n = length(R);

if ~isfield(opts,'abs_tol'),       opts.abs_tol = 1e-6; end
if ~isfield(opts,'rel_tol'),       opts.rel_tol = 1e-6; end
if ~isfield(opts,'max_refine'),    opts.max_refine = 20; end
if ~isfield(opts,'level_factor'),  opts.level_factor = 5; end

% ----------------------------
% initial mesh: m = 1 panel per dimension (uniform)
% ----------------------------
m_scalar = 1;                  % 用一个标量表示“每维 panel 数”
m = m_scalar * ones(n,1);

[Ssave, evals] = simpson_nd(fun, R, L, m);

epssafe = max(opts.abs_tol, opts.rel_tol * abs(Ssave));
eps = epssafe;

flag = false;

for k = 1:opts.max_refine
    % uniform refinement
    m_scalar = 2 * m_scalar;
    m = m_scalar * ones(n,1);

    [Snew, new_evals] = simpson_nd(fun, R, L, m);
    evals = evals + new_evals;

    % Richardson error estimator (Simpson): current_eps = |Snew-Ssave|/15
    diff = abs(Snew - Ssave);
    current_eps = diff / 15;

    eps = max(opts.abs_tol, opts.rel_tol * abs(Snew));

    % 用 current_eps 与 level_factor*epssafe 做“同一量级”判断
    if current_eps < opts.level_factor * epssafe
        flag = true;
        break
    else
        Ssave = Snew;
    end
end

% 如果没达到同一量级，就直接返回最后一次 Snew
if ~flag
    I = Snew;
    info.iter = opts.max_refine;
    info.evals = evals;
    info.err_est = current_eps;
    info.m = m;
    warning('Maximum refinement reached without reaching same-magnitude criterion.');
    return
end

% ----------------------------
% Safeguarding step

len_max = max(L - R);

denom = max(diff, realmin);             
h = (0.9 * eps / denom)^(1/4);

h_original = len_max / m_scalar;     
h_optimal = min(max(h, h_original/2), h_original*(7/8));

m_optimal_scalar = max(1, round(len_max / h_optimal));
m_optimal = m_optimal_scalar * ones(n,1);

[I, opt_evals] = simpson_nd(fun, R, L, m_optimal);
evals = evals + opt_evals;

info.iter = k;                  
info.evals = evals;
info.err_est = current_eps;      
info.m = m_optimal;
end
