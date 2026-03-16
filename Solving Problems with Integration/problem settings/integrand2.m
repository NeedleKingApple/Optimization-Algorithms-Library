function val = integrand2(t, x)
% integrand for f(x) = ∫_Ω ( h1(t,x) + h2(x) ) dt
% t, x are n×1 column vectors
%
% NOTE: p is taken from a global variable P_EXPONENT

global P_EXPONENT;
p = P_EXPONENT;

h1 = exp(t.' * x);                  % exp(sum_i t_i x_i)
idx = (1:length(x)).';              % 1,2,...,n
h2 = sum( (x.^2) ./ (idx.^p) );     % sum_i x_i^2 / i^p

val = h1 + h2;
end
