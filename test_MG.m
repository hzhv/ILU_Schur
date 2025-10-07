%%  k = [1, 2, 8, 16, 32];
% clear;clc;close all;
% n = 100;
% rng("default")
% A = rand(n);
% A = A'*A;     % symmetry
% rhs = (1:n)';

A = load('./A_level2.mat').A;
n = size(A,1);
rhs = load('./rhs_level2.mat').x;
rhs = rhs(:,1);

lb = {};
index = 1;
load("eigs200.mat");
%%
% k = 1;
v = extractEigs(eigs200, 1);
v = orth(v);

tol_inner = 0.3; maxit_inner = 100;
tol_outer = 1e-5; maxit_outer = 30;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 1, 'min');
plotMGD(inner_iter_vec, resvec_outer, 'min'); hold on;
lb{index} = "min res, k=1";
index = index + 1;
%%
[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 1, 'bicgstab');
plotMGD(inner_iter_vec, resvec_outer); hold on;
lb{index} = "bicgstab, k=1";
index = index + 1;
%% =====================================================================================
% k = 2;
v = extractEigs(eigs200, 2);
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
lb{index} = "bicgstab unprec";
index = index + 1;

[sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, 0, 'min');
semilogy(0:length(resvec_outer)-1, resvec_outer, '-o'); hold on;
lb{index} = "min res unprec";
index = index + 1;
% =====================================================================================
% k = 4
v = extractEigs(eigs200, 4);
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

%=====================================================================================
% k = 8;
v = extractEigs(eigs200, 8);
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

%=====================================================================================
% k = 16;
v = extractEigs(eigs200, 16);
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

%=====================================================================================
% k = 32;
v = extractEigs(eigs200, 32);
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

grid on;
legend(lb);
%%
load("eigs200.mat");

d_200 = sort(abs(diag(eigs200{2})), "descend");

semilogy(d_200); grid on;
title("Eigenvalue Magnitude Decay");
xlabel("Magnitude Descend");
ylabel("|Eig Val|")

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

function [v, d] = getEigs(A, k, tol, maxit)
    n = size(A, 1);
    t=@(A,b)bicgstab(A,b,1e-1,100); % -> A^{-1}*x with a precision of 1e-1
    [v, d] = eigs(@(x)t(A',t(A,x)), n, k, 'largestimag', ...
              'Tolerance',tol,'MaxIterations',maxit);
end

function [v, d] = extractEigs(eigs, k)
    if nargin < 2, k = 1; end
    V = eigs{1};
    D = eigs{2};
    d = diag(D);
    [~, perm] = sort(d, "descend"); % compare the real part for C
    permD = D(perm, perm);
    permV = V(:, perm); 
    v = permV(:, 1:k);
    d = permD(1:k, 1:k);
end
