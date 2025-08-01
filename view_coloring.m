%% This script is mainly for visualization of different coloring
clc; clear;

dims = [8,8,8,8];
A = lap_kD_periodic(dims,1);
figure
spy(A)
title("A")
k=1;
N = prod(dims);

% Even-Odd Reordering
[RedBlack, nColors] = distance_1_coloring(A);
[~, perm] = sort(RedBlack);
A_eo = A(perm, perm); % even-odd 
figure
spy(A_eo)
title("A_{perm}: After Even-Odd")

% distance-2
figure
spy(A_eo * A_eo)
title("A_{perm} x A_{perm}")
[colors, nColors] = distance_1_coloring(A_eo * A_eo);

colors_orig       = zeros(N,1);
colors_orig(perm) = colors;           
[~, order_orig]   = sort(colors_orig);
newPos            = zeros(N,1);
newPos(order_orig)= 1:N;


[rowIdx, colIdx] = find(A);
rowIdxNew = newPos(rowIdx);
colIdxNew = newPos(colIdx);

figure;
scatter(colIdxNew, rowIdxNew, 15, colors_orig(rowIdx), 'filled');
axis equal tight;
set(gca, 'YDir','reverse');
colormap(parula(nColors));
colorbar;
title(sprintf('Reordered Matrix (Colors=%d, k=%d)', nColors, k));
xlabel('Col Index after Reordering');
ylabel('Row Index after Reordering');

%% With even-odd ordering
clc; clear;close all; clf;

dims = [8,8,8,8];
k=2;
p = [0 0 0 0];

N = prod(dims);
A = lap_kD_periodic(dims,1);
[colors, nColors] = displacement_even_odd_coloring_nD_lattice(dims, k, p);

[rowIdx, colIdx] = find(spones(A));

% Reording by Colors
[~, order] = sort(colors);     
newPos = zeros(N,1);
newPos(order) = 1:N;          
% A_perm = A(order, order);
% spy(A_perm);

rowIdxNew = newPos(rowIdx);
colIdxNew = newPos(colIdx);

figure;
subplot(1,2,1); spy(A); title("Orignial A");
subplot(1,2,2);
scatter(colIdxNew, rowIdxNew, 15, colors(rowIdx), 'filled');
axis equal tight;
set(gca, 'YDir','reverse');
colormap(parula(nColors));
colorbar;
title(sprintf('Reordered Matrix (Colors=%d, k=%d)', nColors, k));
xlabel('Col Index after Reordering');
ylabel('Row Index after Reordering');

%% Without Even-Odd Reordering  (Fun Curve)
clc; clear; close all; clf;
dims = [8 8 8 8];
k=2;
p = [0 0 0 1];
A = lap_kD_periodic(dims,1);
N = prod(dims);
[colors, nColors] = displacement_coloring_nD_lattice(dims,k,p);
[~, order] = sort(colors);     

A_perm = A(order, order);
figure;spy(A_perm);
newPos = zeros(N,1);
newPos(order) = 1:N;           % newPos(oldIndex) = newIndex

[rowIdx, colIdx] = find(spones(A*A));
rowIdxNew = newPos(rowIdx);
colIdxNew = newPos(colIdx);

figure;
scatter(colIdxNew, rowIdxNew, 15, colors(rowIdx), 'filled');
axis equal tight;
set(gca, 'YDir','reverse');
colormap(parula(nColors));
colorbar;
title(sprintf('Reordered Matrix (Colors=%d, k=%d)', nColors, k));
xlabel('Col Index after Reordering');
ylabel('Row Index after Reordering');
%%
function offs = generate_offsets(d, p)
% Generate the offset that the distance <= p under d dimension
% First enumerate all the offsets within [-p : p], then filter them with p
% Output：
%    offs  M×d matrix，  % each row is an offset and its L1-norm <= p
%                        % M = # of distance<=p neighbors
%                        % e.g. d=4, p=1, then M=9

grid = cell(1,d);
for i=1:d
    grid{i} = -p:p;
end

C = cell(1,d);

% X(i,j,k,l) == grid{1}(i)
% Y(i,j,k,l) == grid{2}(j)
% Z(i,j,k,l) == grid{3}(k)
% T(i,j,k,l) == grid{4}(l)
[C{:}] = ndgrid(grid{:});

for k = 1:d
    C{k} = C{k}(:);  
end

allOffs = [C{:}];   % (2p+1)^d × d

manh = sum(abs(allOffs),2); % Manhattan Distance (L1-norm)
offs = allOffs(manh<=p, :);
end 