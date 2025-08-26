function [Colors, nColors] = displacement_coloring_nD_lattice(D, k, p)
% Input：
%   D = []     % Array of lattice dimension sizes of length d
%   k          % Coloring distance
%   p = []     % Displacement array of length d
%
% Output:
%   Colors     % Array of lattice colors
d = length(D);
N = prod(D);
Colors = zeros(N, 1);

% Get the neighbors that distance <= p
% offs = generate_offsets(size(dims, 2), p);  % size = M x d
offs = Create_Stencil(zeros(1,d),k,p,1);
% natural ordering greedy coloring
for idx = 1:N 
    coords = cell(1,d);
    [coords{:}] = ind2sub(D, idx);
    x = cell2mat(coords) - 1; % 0-based

    % label the colors that neighbors used
    used = false(1, N);
    for j = 1:size(offs,1) % traverse each direction by offset
        dx = offs(j, :);
        if all(dx == 0)
            continue;  % (0, 0, 0, 0)
        end
        % periodically find the ind of each neighbor
        % mod(Index2Coord([1:N],D),D)
        xx = mod(x+dx, D(:)'); % force D to row vector
        new_coords = num2cell(xx+1);
        nbr = sub2ind(D, new_coords{:});
        if nbr < idx  
            used(Colors(nbr)) = true;
        end
    end
    
    % c = find(~used, 1, 'first');
    c=1;
    while used(c)
        c = c + 1;
    end
    
    Colors(idx) = c;
end

nColors = length(unique(Colors));
end
