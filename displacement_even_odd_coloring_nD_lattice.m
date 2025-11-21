%    One important detail for the even-odd experiments is to produce
% compatible coloring schemas with even-odd coloring. That means that all
% the vertices of the same color have to have the same oddity (being
% either even or odd). And you have to make sure that you group the colors
% properly such that when you take half of the matrix, you take all
% vertices being even/odd.

function [Colors, nColors] = displacement_even_odd_coloring_nD_lattice(dim, k, p)
% Input：
%   dim = []     % Array of lattice dimension sizes of length d
%   k            % Coloring distance
%   p = []       % Displacement array of length d
%
% Output:
%   Colors     % coloring results for each vertices
%   nColors    % number of Colors

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
[~, eoPerms] = sort(eo);   % eoPerm is cur ele in original eo position
natOrder(eoPerms) = 1:N;   % Nature Ordering Map

% Get the neighbors that distance <= k
offs = Create_Stencil(zeros(1,d),k,p,1);

maxColorEven = 0;          % To ensure same color has same oddity (parity)
for t = 1:N                % Natural ordering greedy coloring
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
    
    if eo(idx) == 0 % EVEN, color it as usual
        Colors(idx) = c;
        if c > maxColorEven; maxColorEven = c;end
    else            % ODD, add some "shift"
        Colors(idx) = c + maxColorEven;
    end
    nColors = length(unique(Colors));
end
