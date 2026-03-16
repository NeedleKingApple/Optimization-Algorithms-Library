% 把一维求积节点与权重，通过张量积（tensor product）的方式，构造出
% n 维数值积分的所有网格节点及其对应的权重。

function [T,W] = ndgrid_nodes_weights(nodes,weights,n)
% nodes, weights : 1D vectors
% n              : dimension

cell_nodes = cell(1,n);
cell_weights = cell(1,n);

for i = 1:n
    cell_nodes{i} = nodes;
    cell_weights{i} = weights;
end

[grid_nodes{1:n}] = ndgrid(cell_nodes{:});
[grid_weights{1:n}] = ndgrid(cell_weights{:});

num = numel(grid_nodes{1});
T = zeros(num,n);
W = zeros(num,n);

for i = 1:n
    T(:,i) = grid_nodes{i}(:);
    W(:,i) = grid_weights{i}(:);
end
end