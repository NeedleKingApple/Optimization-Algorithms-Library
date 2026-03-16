function val = integrand_grad2(t, x, k)
% k-th component integrand for (g(x))_k
% (g(x))_k = ∫_Ω ( t_k*h1(t,x) + 2*x_k/k^p ) dt
%
% NOTE: p is taken from a global variable P_EXPONENT

global P_EXPONENT;
p = P_EXPONENT;

h1 = exp(t.' * x);
val = t(k) * h1 + (2 * x(k)) / (k^p);
end
