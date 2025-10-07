%%
clear;clc;close all;
% n = 100;
% rng("default")
% A = rand(n);
% A = A'*A;     % symmetry
% rhs = (1:n)';

A = load('./A_level2.mat').A;
n = size(A,1);
rhs = load('./rhs_level2.mat').x;
rhs = rhs(:,1);
% % k = [1, 2, 8, 16, 32];
lb = {};
index = 1;

t=@(A,b)bicgstab(A,b,1e-1,100); % -> A^{-1}*x with a precision of 1e-1
%%
k = 1;
[v, d] = eigs(@(x)t(A',t(A,x)), n, k, 'largestimag', ...
              'Tolerance',0.3,'MaxIterations',30);
fprintf("Eigs done!!!\n")

v = orth(v);
tol_inner = 0.3; maxit_inner = 100;
tol_outer = 1e-5; maxit_outer = 30;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 1, 'min');
plotMGD(inner_iter_vec, resvec_outer, 'min'); hold on;
lb{index} = "min res, k=1";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 1, 'bicgstab');
plotMGD(inner_iter_vec, resvec_outer); hold on;
lb{index} = "bicgstab, k=1";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 0, 'bicgstab');
semilogy(0:length(resvec_outer)-1, resvec_outer, '-o'); hold on;
lb{index} = "bicgstab unprec, k=1";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 0, 'min');
semilogy(0:length(resvec_outer)-1, resvec_outer, '-o'); hold on;
lb{index} = "min res unprec, k=1";
index = index + 1;

% =====================================================================================
k = 2;
[v, d] = eigs(@(x)t(A',t(A,x)), n, k, 'largestimag', ...
              'Tolerance',0.3,'MaxIterations',30);
fprintf("Eigs done!!!\n")
v = orth(v);

tol_inner = 0.3; maxit_inner = 100;
tol_outer = 1e-5; maxit_outer = 30;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 1, 'min');
plotMGD(inner_iter_vec, resvec_outer, 'min'); hold on;
lb{index} = "min res, k=2";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 1, 'bicgstab');
plotMGD(inner_iter_vec, resvec_outer, 'bicgstab'); hold on;
lb{index} = "bicgstab, k=2";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 0, 'bicgstab');
semilogy(0:length(resvec_outer)-1, resvec_outer, '-o'); hold on;
lb{index} = "bicgstab unprec, k=2";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 0, 'min');
semilogy(0:length(resvec_outer)-1, resvec_outer, '-o'); hold on;
lb{index} = "min res unprec, k=2";
index = index + 1;
% =====================================================================================
k = 4;
[v, d] = eigs(@(x)t(A',t(A,x)), n, k, 'largestimag', ...
              'Tolerance',0.3,'MaxIterations',30);
fprintf("Eigs done!!!\n")
v = orth(v);

tol_inner = 0.3; maxit_inner = 100;
tol_outer = 1e-5; maxit_outer = 30;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 1, 'min');
plotMGD(inner_iter_vec, resvec_outer, 'min'); hold on;
lb{index} = "min res, k=4";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 1, 'bicgstab');
plotMGD(inner_iter_vec, resvec_outer, 'bicgstab'); hold on;
lb{index} = "bicgstab, k=4";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 0, 'bicgstab');
semilogy(0:length(resvec_outer)-1, resvec_outer, '-o'); hold on;
lb{index} = "bicgstab unprec, k=4";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 0, 'bicgstab');
semilogy(0:length(resvec_outer)-1, resvec_outer, '-o'); hold on;
lb{index} = "min res unprec, k=4";
index = index + 1;
%=====================================================================================
k = 8;
[v, d] = eigs(@(x)t(A',t(A,x)), n, k, 'largestimag', ...
              'Tolerance',0.3,'MaxIterations',30);
fprintf("Eigs done!!!\n")
v = orth(v);

tol_inner = 0.3; maxit_inner = 100;
tol_outer = 1e-5; maxit_outer = 30;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 1, 'min');
plotMGD(inner_iter_vec, resvec_outer, 'min'); hold on;
lb{index} = "min res, k=8";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 1, 'bicgstab');
plotMGD(inner_iter_vec, resvec_outer, 'bicgstab'); hold on;
lb{index} = "bicgstab, k=8";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 0, 'bicgstab');
semilogy(0:length(resvec_outer)-1, resvec_outer, '-o'); hold on;
lb{index} = "bicgstab unprec, k=8";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 0, 'min');
semilogy(0:length(resvec_outer)-1, resvec_outer, '-o'); hold on;
lb{index} = "min res unprec, k=8";
index = index + 1;

%=====================================================================================
k = 16;
[v, d] = eigs(@(x)t(A',t(A,x)), n, k, 'largestimag', ...
              'Tolerance',0.3,'MaxIterations',30);
fprintf("Eigs done!!!\n")
v = orth(v);

tol_inner = 0.3; maxit_inner = 100;
tol_outer = 1e-5; maxit_outer = 30;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 1, 'min');
plotMGD(inner_iter_vec, resvec_outer, 'min'); hold on;
lb{index} = "min res, k=16";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 1, 'bicgstab');
plotMGD(inner_iter_vec, resvec_outer, 'bicgstab'); hold on;
lb{index} = "bicgstab, k=16";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 0, 'bicgstab');
semilogy(0:length(resvec_outer)-1, resvec_outer, '-o'); hold on;
lb{index} = "bicgstab unprec, k=16";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 0, 'min');
semilogy(0:length(resvec_outer)-1, resvec_outer, '-o'); hold on;
lb{index} = "min res unprec, k=16";
index = index + 1;

%=====================================================================================
k = 32;
[v, d] = eigs(@(x)t(A',t(A,x)), n, k, 'largestimag', ...
              'Tolerance',0.3,'MaxIterations',30);
fprintf("Eigs done!!!\n")
v = orth(v);

tol_inner = 0.3; maxit_inner = 100;
tol_outer = 1e-5; maxit_outer = 30;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 1, 'min');
plotMGD(inner_iter_vec, resvec_outer, 'min'); hold on;
lb{index} = "min res, k=32";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 1, 'bicgstab');
plotMGD(inner_iter_vec, resvec_outer, 'bicgstab'); hold on;
lb{index} = "bicgstab, k=32";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 0, 'min');
semilogy(0:length(resvec_outer)-1, resvec_outer, '-o'); hold on;
lb{index} = "min res unprec, k=32";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 0, 'bicgstab');
semilogy(0:length(resvec_outer)-1, resvec_outer, '-o'); hold on;
lb{index} = "bicgstab unprec, k=32";

grid on;
legend(lb);
%%
function plotMGD(inner_iter_vec, resvec_outer, solver)
    if nargin < 3 || isempty(solver) || strcmpi(solver, 'bicgstab')
        m = "-";
    else
        m = "--";
    end
    resvec_outer = resvec_outer(2:end);
    assert(size(inner_iter_vec,1) == size(resvec_outer,1))
    X = cumsum(inner_iter_vec);
    semilogy(X, resvec_outer, LineStyle=m);
end