function [colors, nColors] = distance_k_coloring_min(D, k)
% Implicitly do the distance-k (A^k) coloring on 4-d periodic sparse matrix
%
% Input：
%   D    = [Lx, Ly, Lz, Lt]   % coordinates of lattice
%   k    = non-negative int   % distance-k 
% Output：
%   colors  N×1 vector        % each point assigned，start from 1
%   nColors int               % total colors assigned
d = numel(D);
Lx=D(1); Ly=D(2); Lz=D(3); Lt=D(4);
N = prod(D);
colors = zeros(N,1); 

% Get the neighbors that distance <= k
offs = generate_offsets(d, k);  % size = M x d

% natural ordering greedy coloring
for idx = 1:N
    [x,y,z,t] = ind2sub(D, idx);
    x=x-1; y=y-1; z=z-1; t=t-1;  % 0-based

    % label the colors that neighbors used
    used = false(1, N);
    for index = 1:size(offs,1) % traverse each direction by offset
        dx=offs(index,1); dy=offs(index,2);
        dz=offs(index,3); dt=offs(index,4);
        if dx==0 && dy==0 && dz==0 && dt==0
            continue;  % (0, 0, 0, 0)
        end
        % periodically find the ind of each neighbor
        % mod(Index2Coord([1:N],D)+k,D)
        xx = mod(x+dx, Lx);
        yy = mod(y+dy, Ly);
        zz = mod(z+dz, Lz);
        tt = mod(t+dt, Lt);
        nbr = sub2ind(D, xx+1, yy+1, zz+1, tt+1);
        if nbr < idx  
            used(colors(nbr)) = true;
        end
    end
    
    % c = find(~used, 1, 'first');
    c=1;
    while used(c)
        c = c + 1;
    end
    
    colors(idx) = c;
end

nColors = max(colors);
end % of coloring

%%
function offs = generate_offsets(d, k)
% Generate the offset that the distance <= k under d dimension
% First enumerate all the offsets within [-k : k], then filter them with k
% Input:
%    d     = len(D), int
%    k     = distance-k, int   
% Output：
%    offs  = M×d matrix，  % each row is an offset and its L1-norm <= k
%                          % M = # of distance<=k neighbors
%                          % e.g. d=4, k=1, then M=9

grid = cell(1,d);
for i=1:d
    grid{i} = -k:k; % jumps
end

C = cell(1,d);

% X(i,j,k,l) == grid{1}(i)
% Y(i,j,k,l) == grid{2}(j)
% Z(i,j,k,l) == grid{3}(k)
% T(i,j,k,l) == grid{4}(l)
[C{:}] = ndgrid(grid{:});

for j = 1:d
    C{j} = C{j}(:);  
end

allOffs = [C{:}];   % (2k+1)^d × d

manh = sum(abs(allOffs),2); % Manhattan Distance (L1-norm)
offs = allOffs(manh <= k, :);
end 

%% Distance-1 Coloring
function [color, nColor] = distance_1_coloring(A)
n = size(A,1);
color = zeros(n,1);
nColor = 0; % total # of colors
for v = 1:n
    nbr = find(A(v,:)); % nnz neighbours of v
    used = color(nbr);
    c = 1;
    while any(used == c)
        c = c + 1;
    end
    color(v) = c;
    if c > nColor, nColor = c; end 

end
end % of coloring