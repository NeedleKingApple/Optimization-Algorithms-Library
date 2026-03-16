function [nodes, weights] = simpson_1d_nodes_weights(m)
% Composite Simpson on [0, 2m] index grid
% nodes   : 0:2m
% weights : [1,4,2,4,...,2,4,1]

nodes = 0:(2*m);
weights = 2*ones(1,2*m+1);    % start with all 2's
weights(2:2:end-1) = 4;       % odd interior indices -> 4
weights(1) = 1;               % endpoints -> 1
weights(end) = 1;
end

% 根据节点的个数，自动编号一维时的索引和对应的权重