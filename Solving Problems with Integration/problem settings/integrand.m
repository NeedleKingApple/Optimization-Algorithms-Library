function val = integrand(t, x)
% 被积函数 (h2(t,x) - h1(t))^2
% t, x : n×1 向量

n = length(t);

s1 = 0;
s2 = 0;
for i = 1:n
    s1 = s1 + t(i)/i;
    s2 = s2 + t(i)/x(i);
end

h1 = exp(-s1);
h2 = cos(s2);

val = (h2 - h1)^2;
end
