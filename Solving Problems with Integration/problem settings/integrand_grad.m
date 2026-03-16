function val = integrand_grad(t, x, k)
% 梯度第 k 个分量的被积函数
% t, x : n×1 向量
% k    : 分量索引

n = length(t);

s1 = 0;
s2 = 0;
for i = 1:n
    s1 = s1 + t(i)/i;
    s2 = s2 + t(i)/x(i);
end

h1 = exp(-s1);
h2 = cos(s2);

val = 2 * (h2 - h1) * sin(s2) * t(k) / (x(k)^2);
end
