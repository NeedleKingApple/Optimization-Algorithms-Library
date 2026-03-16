function [S, evals] = simpson_nd(fun, R, L, m)
% n-D composite Simpson tensor product
% fun : function handle, fun(t) with t n×1
% R,L : bounds (n×1)
% m   : number of Simpson panels per dimension (n×1 or scalar)

n = length(R);

if isscalar(m)
    m = m*ones(n,1);
else
    m = m(:);
end

% step per dimension
h = (L - R) ./ (2*m);

% build 1D nodes/weights per dimension (indices 0..2m_i)
cell_nodes = cell(1,n);
cell_wts   = cell(1,n);
for i = 1:n
    [nodes_i, w_i] = simpson_1d_nodes_weights(m(i));
    cell_nodes{i} = nodes_i;
    cell_wts{i}   = w_i;
end

% tensor grid of indices + tensor grid of weights
grid_nodes = cell(1,n);
grid_wts   = cell(1,n);
[grid_nodes{:}] = ndgrid(cell_nodes{:});
[grid_wts{:}]   = ndgrid(cell_wts{:});

num = numel(grid_nodes{1});
S = 0;
evals = 0;

for k = 1:num
    idx = zeros(n,1);
    wt  = 1;
    for i = 1:n
        idx(i) = grid_nodes{i}(k);
        wt     = wt * grid_wts{i}(k);
    end
    t = R + h .* idx;
    S = S + wt * fun(t);
    evals = evals + 1;
end

% scaling: prod(h)/3^n
S = S * prod(h) / (3^n);
end
