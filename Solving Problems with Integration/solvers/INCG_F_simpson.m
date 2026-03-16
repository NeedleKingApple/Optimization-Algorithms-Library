function out = INCG_F_simpson(fun, gradfun, x0, R, L, opts)
% INCG_simpson: Inexact Nonlinear Conjugate Gradient with Simpson-based inexact oracle
%
% fun(t,x)      : integrand of f(x) = \int fun(t,x) dt
% gradfun(t,x,k): integrand of (∂f/∂x_k) = \int gradfun(t,x,k) dt
% x0            : initial point
% R,L           : integration bounds
% opts          : parameters
%
% Required fields in opts:
%   opts.tau, opts.lambda, opts.delta, opts.itermax
%   opts.simpson_f, opts.simpson_g
%   opts.stop_tol_g
%
% INCG-specific:
%   opts.sigma      in (0,1) for Armijo
%   opts.nu         > 0
%   opts.bar_gamma  >= 1
%   opts.beta_type  = 'HS'/'PRP'/'FR'/'DY' (default 'HS')
%
% Output:
%   out.x, out.f, out.gnorm, out.iter, out.like

x = x0;
n = length(x);

% line search / outer iteration
tau     = opts.tau;
lambda  = opts.lambda;
sigma   = opts.sigma;     % Armijo parameter in (0,1)
itermax = opts.itermax;

% INCG scaling/truncation parameters
nu        = opts.nu;
bar_gamma = opts.bar_gamma;  % >= 1

stop_tol_g = opts.stop_tol_g;

beta_type = 'HS';
if isfield(opts,'beta_type'), beta_type = upper(opts.beta_type); end

like = zeros(itermax,1);

% ----- helpers: Simpson options for f and g -----
opts_f0 = opts.simpson_f;   % function value oracle options (baseline)
opts_g  = opts.simpson_g;   % gradient oracle options

% initial f(x0)
f = simpson_nd_adaptive(@(t) fun(t,x), R, L, opts_f0);

% initial gradient gtilde_0
gtilde = zeros(n,1);
for j = 1:n
    gtilde(j) = simpson_nd_adaptive(@(t) gradfun(t,x,j), R, L, opts_g);
end

gnorm = norm(gtilde);
like(1) = gnorm;

% initial direction
d_prev = -gtilde;
gtilde_prev = gtilde;
x_prev = x;

if gnorm < stop_tol_g
    out.x = x;
    out.f = f;
    out.gnorm = gnorm;
    out.iter = 0;
    out.like = like(1);
    return
end

for k = 1:itermax

    % record
    like(k) = gnorm;

    if gnorm < stop_tol_g
        break
    end

    % -------------------------
    % build d_k from d_{k-1}, gtilde_k
    % -------------------------
    if k == 1
        % already have d_prev = -gtilde_0; treat as d_{0}
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
        beta = compute_beta(beta_type, gtilde, gtilde_prev, ytilde, d_prev);

        % truncation bound bar_beta_k
        if norm(d_h) == 0
            bar_beta = inf;
        else
            % requires nu*bar_gamma >= 1 to make sqrt nonnegative
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
    % enforce gamma_k <= bar_gamma implicitly via theory (but numerically we can clamp)
    gamma_k = min(gamma_k, bar_gamma);

    p_k = d_k / gamma_k;

    % ensure descent w.r.t. gtilde (should hold: gtilde^T p_k = -(1/gamma_k)||gtilde||^2)
    % but numerically you might want a safeguard:
    % % if gtilde.'*p_k >= 0
    % %    fallback to steepest descent direction
    %     p_k = -gtilde;
    %     gamma_k = 1;
    % end

    % -------------------------
    % Armijo-type inexact line search (fixed initial step tau)
    % -------------------------
    alpha = tau;
    f0 = f;

    % optional: you can use a decaying absolute tol for f in linesearch
    % to emulate your IGD schedule: tol_f = opts.simpson_f.abs_tol / max(1,k)^2
    opts_f = opts.simpson_f;
    if isfield(opts,'tol_f0') && ~isempty(opts.tol_f0)
        opts_f.abs_tol = opts.tol_f0 / max(1,k)^2;
    end
    tol_ls = opts_f.abs_tol * 5;
    while true
        x_trial = x + alpha * p_k;
        fnew = simpson_nd_adaptive(@(t) fun(t,x_trial), R, L, opts_f);

        % Armijo (inexact): f(x+ap) <= f(x) + sigma*a*gtilde^T p
        if fnew <= f0 + sigma*alpha*(gtilde.'*p_k) + tol_ls
            break
        end

        alpha = lambda * alpha;
        if alpha < 1e-16
            % avoid infinite loop
            break
        end
    end

    % update
    x_prev = x;
    x = x + alpha * p_k;
    f = fnew;

    % update gradient
    gtilde_prev = gtilde;
    gtilde = zeros(n,1);
    for j = 1:n
        gtilde(j) = simpson_nd_adaptive(@(t) gradfun(t,x,j), R, L, opts_g);
    end
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


% ===========================
% beta computation helper
% ===========================
function beta = compute_beta(beta_type, gk, gkm1, yk, dkm1)
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
        % Hager–Zhang (HZ) beta:
        % beta = (y - 2 d * ||y||^2/(d^T y))^T gk / (d^T y)
        denom = dkm1.' * yk;              % d_{k-1}^T y_k
        if abs(denom) < 1e-16
            beta = 0;
        else
            y_norm2 = yk.' * yk;          % ||y_k||^2
            y_hat   = yk - 2 * dkm1 * (y_norm2 / denom);
            beta    = (y_hat.' * gk) / denom;
        end

    otherwise
        error('Unknown beta_type: %s', beta_type);
end
end
