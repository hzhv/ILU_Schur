function Stencil = Create_Stencil(x, k, p, dim)
% Input:
%   x = d-dimensional array to store an offset, row-vector;
%   k = Coloring distance, int;
%   p = Displacement array of length d, row-vector;
%   dim = Recursion/dimension level;
%
% Output:
%   Stencil = A mapping of a lattice coordinate’s neighbors, M x d array;

if dim == 1
   Stencil = []; 
end

    if dim > numel(x)  % [0 0 0 0]
        rows = [x + p; % [0 0 10 0]
                x - p];
        Stencil = unique(rows, 'rows');
        return;
    end

    % 递归构造：第 dim 维从 -k 到 +k
    Stencil = [];  % 本层先清空
    for j = -k : k
        x(dim) = j;
        S = Create_Stencil(x, k - abs(j), p, dim + 1);
        Stencil = [Stencil; S];   % 垂直拼接所有子结果
    end

    % 每层去重一次，避免中间爆炸
    Stencil = unique(Stencil, 'rows');
end



%%
% function offs = Create_Stencil(k, p, d)
% % Generate the offset that the distance <= p under d dimension
% % First enumerate all the offsets within [-p : p], then filter them with p
% % Input:
% %    d     = dimension int
% %    p     = non-negative int   % distance-p 
% % Output：
% %    offs  = M×d matrix，  % each row is an offset and its L1-norm <= p
% %                          % M = # of distance<=p neighbors
% %                          % e.g. d=4, p=1, then M=9
% 
% grid = cell(1,d);
% for i=1:d
%     grid{i} = -p:p; % jumps
% end
% 
% C = cell(1,d);
% 
% % X(i,j,k,l) == grid{1}(i)
% % Y(i,j,k,l) == grid{2}(j)
% % Z(i,j,k,l) == grid{3}(k)
% % T(i,j,k,l) == grid{4}(l)
% [C{:}] = ndgrid(grid{:});
% 
% for k = 1:d
%     C{k} = C{k}(:);  
% end
% 
% allOffs = [C{:}];   % (2p+1)^d × d
% 
% manh = sum(abs(allOffs),2); % Manhattan Distance (L1-norm)
% offs = allOffs(manh <= k, :);
% end 