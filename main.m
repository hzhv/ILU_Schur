% 08/011/2025
%  Now support Kron to make more "boxes" at diagonals
%% Testing the followings:
%
%   Permute matrix with multicoloring, 
%   apply ILU(0) preconditioner, 
%   employ even-odd reordering, 
%   implementing the Schur complement for solving Ax = b.
%
%  Main args:
%    D       : length n vector for generating nd lattice
%    p       : length n vector for displacement at each d
%    k_total : 1:k distance, i.e. A^k
%   
%    A       : NxN Sparse matrix (square, nonsingular)
%    Colors  : Nx1 permutation matrix 
%    b       : Right-hand side vector
%    tol     : tolerance for iterative solves

clc; clear; close all;
D = [8 8 8 16];
p = [0 0 0 0];
k_total = 3;
tol = 1e-6;

A = lap_kD_periodic(D,1); % eigen vector all the same % this on GPU
rng(1);
A = kron(A, (rand(48))); B_perm = ones(48,1); % For denser "boxed" diag
N_new = size(A, 1);
A = A + speye(N_new)*1e-7; % non-singular

% A = A.*(1+rand(N,N)/10) + speye(N)*1e-7;
% figure; spy(A); title("A Original")

% b = ones(size(A,1), 1);  % RHS: all-ones vector
b = randn(N_new, 1);
%%
maxit = 1000;
restart = 200;
% x0 = b;
%% "Pure Iterations"
[x, flag, relres, iter, resvec] = ... % 
    gmres(A, b, restart, tol, maxit, [], []);
relres_true = norm(b - A*x)/norm(b);
fprintf('Pure iterative results w/o preconditioner:\n');
fprintf('  The actual residual norm = %d\n', relres_true);
fprintf('  GMRES projection relative residual %e in %d iterations.\n\n', relres, sum(iter));
%% Use ILU(0) Preconditioner Only
fprintf('Use ilu(0) but no multi-reordering:\n');
setup.type    = 'nofill';
setup.droptol = 0;  % exact ILU(0)
[L, U] = ilu(A, setup);
M_handle = @(x) U\(L\x);

maxit = size(A,1);
restart = size(A, 1);
x0 = b;
[x_perm, flag, relres, iter, resvec] = ...
    gmres(A, b, restart, tol, maxit, [], M_handle, x0);
relres_true = norm(b - A*x_perm)/norm(b);

fprintf('  The actual residual norm = %d\n', relres_true);
fprintf('  GMRES projection relative residual %e in %d iterations.\n\n', relres, sum(iter));

%% Only muticoloring without Even-Odd ordering
fprintf('Only multi-coloring w/o Even-Odd:\n');
iters = zeros(1, k_total);
relres_true = cell(1, k_total);

for k = 1:k_total
[Colors, nColors] = displacement_coloring_nD_lattice(D, k, p);
Colors = kron(Colors, B_perm);
[~, perm] = sort(Colors);
A_perm = A(perm, perm);
b_perm = b(perm);

setup.type    = 'nofill';
setup.droptol = 0;  % exact ILU(0)
[L, U] = ilu(A_perm, setup);
M_handle = @(x) U\(L\x);
% M1 = eye(N);

maxit = size(A_perm,1);
restart = size(A_perm, 1);
% x0 = b_perm;

[x_perm, flag, relres, iter, resvec] = ...
    gmres(A_perm, b_perm, restart, tol, maxit, [], M_handle);

relres_true_ = norm(b_perm - A_perm*x_perm)/norm(b_perm);

if flag == 0
    fprintf('  When k = %d,', k);
    fprintf('  total colors = %d.\n', nColors);
    fprintf('  the actual residual norm = %d\n', relres_true_);
    fprintf('  GMRES projection relative residual %e in %d iterations.\n\n', relres, sum(iter));
    
else
    fprintf('  GMRES failed to converge (flag = %d). Relative residual = %e.\n', flag, relres);
end
iters(k) = sum(iter);
relres_true{k} = resvec/norm( U\b_perm ); 
end
%% Multi-reordering w/ Even-Odd Reordering
fprintf('Even-Odd reordering then multi-coloring:\n');
iters_eo = zeros(1, k_total);
relres_true_eo = cell(1, k_total);

for k = 1:k_total
% [Colors, nColors] = displacement_coloring_nD_lattice(D, k, p);
[Colors, nColors] = displacement_even_odd_coloring_nD_lattice(D, k, p);
Colors = kron(Colors, B_perm);
[~, perm] = sort(Colors);
A_perm = A(perm, perm);
b_perm = b(perm);
% colorView(A_perm,Colors, nColors, k);

setup.type    = 'nofill';
setup.droptol = 0;  % exact ILU(0)
[L, U] = ilu(A_perm, setup);
M_handle = @(x) U\(L\x);

maxit = size(A_perm,1);
restart = size(A_perm, 1);
% x0 = zeros(size(b_perm, 1),1);
% x0 = b_perm;

[x_perm, flag, relres, iter, resvec] = ...
    gmres(A_perm, b_perm, restart, tol, maxit, [], M_handle);

relres_true_ = norm(b_perm - A_perm*x_perm)/norm(b_perm);

if flag == 0
    fprintf('  When k = %d,', k);
    fprintf('  total colors = %d.\n', nColors);
    fprintf('  the actual residual norm = %d\n', relres_true_);
    fprintf('  GMRES projection relative residual %e in %d iterations.\n\n', relres, sum(iter));

else
    fprintf('  GMRES failed to converge (flag = %d). Relative residual = %e.\n', flag, relres);
end
iters_eo(k) = sum(iter);
relres_true_eo{k} = resvec/norm( U\b_perm );
end
% Undo permutation to original order
% invperm(perm) = 1:length(perm);
% x = x_perm(invperm);


%% Partial ILU(0)(A) with Schur Complement
fprintf('Schur Complement:\n')
n = N_new;
% M = l*u; 
% M_ee=M(1:n/2,1:n/2); %
resvec_even = cell(1, k_total);
for k = 1:k_total
    [colors, nColors] = displacement_coloring_nD_lattice(D, k, p);
    % [colors, nColors] = displacement_even_odd_coloring_nD_lattice(D, k, p);
    colors = kron(colors, B_perm);
    [~, perm] = sort(colors);
    A_perm = A(perm, perm);
    b_perm = b(perm);
    colorView(A_perm, colors, nColors, k);
    
    % Preconditioner
    setup.type    = 'nofill';
    setup.droptol = 0; 
    [L, U] = ilu(A_perm, setup);
    M = L*U;
    M = M(1:n/2, 1:n/2); % M_ee
    
    ap_ee = A_perm(1:n/2,1:n/2);
    ap_oo = A_perm(n/2+1:end,n/2+1:end);
    %%%% Inverse ap_oo in advance
    ap_oo_inv = inv(ap_oo);
    % r = rank(full(ap_oo));
    % fprintf("rank(ap_oo) = %d, size = %d\n", r, size(ap_oo,1));
    ap_eo = A_perm(1:n/2, n/2+1:end);
    ap_oe = A_perm(n/2+1:end, 1:n/2);
    
    bp_e = b_perm(1:n/2);
    bp_o = b_perm(n/2+1 : end);

    % rhs_o = bp_o - ap_oe * (ap_ee \ bp_e);
    rhs_e = bp_e - ap_eo * (ap_oo \ bp_o);
    
    % Eliminate odd -> GMRES solve for even
    maxit = size(ap_ee,1);
    restart = size(ap_ee, 1);

    tic;
    s_ee_mul = ap_ee - ap_eo * (ap_oo \ ap_oe);         % EXPLICITLY
    [x_even, flag, relres_even, iter_even, resvec_even{k}] = ...
        gmres(s_ee_mul, rhs_e, restart, tol, maxit, [], M);
    t_exp = toc;
    fprintf('  GMRES w/ exp Schur used %d sec\n', t_exp);

    % figure; clf; spy(s); title(sprintf('S_{even} for k=%d', k)); grid on; 
    tic;
    s_ee_mul = @(x)ap_ee*x - ap_eo * (ap_oo \ (ap_oe*x)); % IMPLICITLY, do the inv ap_oo in advance, inv(ap_oo) should be also diag not dense.
    
    % s_ee_mul = @(x)ap_ee*x - ap_eo * (ap_oo_inv*(ap_oe*x)); 
    % x0 = bp_e_t;
    [x_even, flag, relres_even, iter_even, resvec_even{k}] = ...
        gmres(s_ee_mul, rhs_e, restart, tol, maxit, [], M);
    t_imp = toc;
    fprintf('  When k = %d,', k)
    fprintf('  Total colors = %d\n', nColors);
    fprintf('  GMRES projection relative residual %e in %d iterations.\n', relres_even, sum(iter_even));
    fprintf('  GMRES w/ imp Schur used %d sec\n\n', t_imp);
    x_odd = ap_oo \ (bp_o - ap_oe * x_even);
    % Combine x_even and x_odd
end

figure; clf;
for k = 1:k_total
    semilogy(resvec_even{k}, '-', 'LineWidth', 1.5, 'DisplayName', sprintf('w/ EO, k = %d', k));
    hold on;
end
xlabel('Iteration');
ylabel('Residual Norm');
yline(tol,'r--','DisplayName', sprintf('Tol'));
title('Schur Comp. GMRES Residual Convergence without EO');
legend show;
grid on;
saveas(gcf, 'Schur_Res.png')
%%
function colorView(A, colors, nColors, k)
figure;
N = size(A, 1);
% Reording by Colors
[~, order] = sort(colors);     
newPos = zeros(N,1);
newPos(order) = 1:N;          
[rowIdx, colIdx] = find(spones(A));
rowIdxNew = newPos(rowIdx);
colIdxNew = newPos(colIdx);


subplot(1,2,1); spy(A); title(sprintf("A^{%g}", k));
subplot(1,2,2);
scatter(colIdxNew, rowIdxNew, 15, colors(rowIdx), 'filled');
axis equal tight;
set(gca, 'YDir','reverse');
colormap(parula(nColors));
colorbar;
title(sprintf('Reordered Matrix (Colors=%d, k=%d)', nColors, k));
xlabel('Col Index after Reordering');
ylabel('Row Index after Reordering');
grid on;
end