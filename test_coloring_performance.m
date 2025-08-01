%% Permute matrix, apply ILU(0) preconditioner, and solve Ax = b
% Inputs:
% A       : NxN Sparse matrix (square, nonsingular)
% Colors  : Nx1 permutation matrix 
% b       : Right-hand side vector

clc; clear; close all;
D = [8 8 8 8];
p = [1 0 0 0];
k_total = 2; % Distance, 1:k_total
tol = 0.3;

A = lap_kD_periodic(D,1);
N = size(A,1);
A = A + speye(N)*1e-7;
figure; spy(A); title("A Original")

b = ones(size(A,1), 1);  % RHS: all-ones vector

maxit = size(A,1);
restart = size(A, 1);
x0 = b;

[x, flag, relres, iter, resvec] = ...
    gmres(A, b, restart, tol, maxit, [], [], x0);
relres_true = norm(b - A*x)/norm(b);
fprintf('Pure iterative results w/o preconditioner:\n');
fprintf('  The actual residual norm = %d\n', relres_true);
fprintf('  GMRES projection relative residual %e in %d iterations.\n\n', relres, sum(iter));
%% Use ILU(0) Preconditioner
fprintf('Use ilu(0) but no multi-reordering:\n');
setup.type    = 'nofill';
setup.droptol = 0;  % exact ILU(0)
[L, U] = ilu(A, setup);

maxit = size(A,1);
restart = size(A, 1);
x0 = b;
[x_perm, flag, relres, iter, resvec] = ...
    gmres(A, b, restart, tol, maxit, L, U, x0);
relres_true = norm(b - A*x)/norm(b);

fprintf('  The actual residual norm = %d\n', relres_true);
fprintf('  GMRES projection relative residual %e in %d iterations.\n\n', relres, sum(iter));
%% Only muticoloring without Even-Odd ordering
fprintf('Only multi-coloring w/o Even-Odd:\n');
iters = zeros(1, k_total);
relres_true = cell(1, k_total);

for k = 1:k_total
[Colors, nColors] = displacement_coloring_nD_lattice(D, k, p);
[~, perm] = sort(Colors);
A_perm = A(perm, perm);
b_perm = b(perm);

setup.type    = 'nofill';
setup.droptol = 0;  % exact ILU(0)
[L, U] = ilu(A_perm, setup);
% M1 = eye(N);

maxit = size(A_perm,1);
restart = size(A_perm, 1);
x0 = b_perm;

[x_perm, flag, relres, iter, resvec] = ...
    gmres(A_perm, b_perm, restart, tol, maxit, [], U, x0);

relres_true_ = norm(b_perm - A_perm*x_perm)/norm(b_perm);

if flag == 0
    fprintf('  When k = %d,', k)
    fprintf('  the actual residual norm = %d\n', relres_true_);
    fprintf('  GMRES projection relative residual %e in %d iterations.\n\n', relres, sum(iter));
else
    fprintf('  GMRES failed to converge (flag = %d). Relative residual = %e.\n', flag, relres);
end
iters(k) = sum(iter);
relres_true{k} = resvec/norm( U\b_perm );
end
%% Multi-reordering: Muticoloring included Even-Odd Reordering
fprintf('Even-Odd reordering then multi-coloring:\n');
iters_eo = zeros(1, k_total);
relres_true_eo = cell(1, k_total);

for k = 1:k_total
% [Colors, nColors] = displacement_coloring_nD_lattice(D, k, p);
[Colors, nColors] = displacement_even_odd_coloring_nD_lattice(D, k, p);
[~, perm] = sort(Colors);
A_perm = A(perm, perm);
b_perm = b(perm);
figure;spy(A_perm);
setup.type    = 'nofill';
setup.droptol = 0;  % exact ILU(0)
[L, U] = ilu(A_perm, setup);

maxit = size(A_perm,1);
restart = size(A_perm, 1);
x0 = b_perm;

[x_perm, flag, relres, iter, resvec] = ...
    gmres(A_perm, b_perm, restart, tol, maxit, [], U, x0);

relres_true_ = norm(b_perm - A_perm*x_perm)/norm(b_perm);

if flag == 0
    fprintf('  When k = %d,', k)
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
%% Vis
figure;
plot(1:k_total, iters, '-o', 'LineWidth', 2); hold on;
plot(1:k_total, iters_eo, '-s', 'LineWidth', 2);
xlabel('k'); ylabel('GMRES Iterations');
legend('Without EO', 'With EO','Location', 'northwest');
title('GMRES Iterations vs. k');
grid on;

figure;
hold on;
for k = 1:k_total
    semilogy(relres_true{k},'--' ,'LineWidth', 1.2, 'DisplayName', sprintf('w/o EO, k = %d', k));
end
% xlabel('Iteration');
% ylabel('Residual Norm');
% yline(tol,'r--');
% title('GMRES Residual w/o EO');
% legend show;
% grid on;

% figure;
% hold on;
for k = 1:k_total
    semilogy(relres_true_eo{k},'-', 'LineWidth', 1.2, 'DisplayName', sprintf('w/ EO, k = %d', k));
end
xlabel('Iteration');
ylabel('Residual Norm');
yline(tol,'r--','DisplayName', sprintf('Tol'));
title('GMRES Residual Convergence (With vs. Without EO)');
legend show;
grid on;