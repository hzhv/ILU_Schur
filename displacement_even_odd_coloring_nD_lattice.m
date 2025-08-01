function [Colors, nColors] = displacement_even_odd_coloring_nD_lattice(D, k, p)
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
natOrder = zeros(N, 1);
eo = zeros(N,1); % Even-odd (black-red) reordering
for i = 1:N
    coords = cell(1, d);
    [coords{:}] = ind2sub(D, i);
    x = cell2mat(coords) - 1;
    eo(i) = mod(sum(x),2);
end
[~, eoPerms] = sort(eo);
natOrder(eoPerms) = 1:N; % Nature Ordering Map

% Get the neighbors that distance <= k
offs = Create_Stencil(zeros(1,d),k,p,1);

% Natural ordering greedy coloring
for t = 1:N 
    idx = eoPerms(t);
    coords = cell(1,d);
    [coords{:}] = ind2sub(D, idx);
    x = cell2mat(coords) - 1; % 0-based

    % label the colors that neighbors used
    used = false(1, N);
    for k = 1:size(offs,1) % traverse each direction by offset
        dx = offs(k, :);
        if all(dx == 0)
            continue;  % (0, 0, 0, 0)
        end
        % periodically find the ind of each neighbor
        % mod(Index2Coord([1:N],D),D)
        xx = mod(x+dx, D(:)'); % force D to row vector
        new_coords = num2cell(xx+1);
        nbr = sub2ind(D, new_coords{:});
        if natOrder(nbr) < natOrder(idx) % Back to Nature Ordering
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

nColors = max(Colors);
end
