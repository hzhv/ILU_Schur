%% Note these tests only for plotting

% u = 1:max(iters)
% interp1(iters, resvec, [1:max(iters)], 'linear');
% trunc by minimum 


clear; clc; close all;
% n = 100;
% rng("default")
% A = rand(n);
% A = A'*A;     % symmetry
% rhs = (1:n)';

A = load('./A_level2.mat').A;
n = size(A,1);
rhs = load('./rhs_level2.mat').x;
% rhs = rhs(:,1); %% try more rhs plz
% [v, d] = getEigs(A, 200, 0.01, 1000);

setup.type    = 'nofill';
setup.droptol = 0;  
[L, U] = ilu(A, setup);
M_smo_ilu0 = @(x) U\(L\x);

tol_inner = 0.01; maxit_inner = 4;
tol_outer = 1e-2; maxit_outer = 30;

lb = {};
index = 1;
eigs200 = load("eigs200_1e1_1000.mat").eigCell;
%%
v = extractEigs(eigs200, 1);
v = orth(v);

test_unprec( ...
    A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    0, M_smo_ilu0, 'bicgstab');
lb{index} = "bicgstab unprec";
index = index + 1;

test_unprec( ...
    A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    0, M_smo_ilu0, 'min');
lb{index} = "min res unprec";
index = index + 1;

test_unprec( ...
    A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    2, M_smo_ilu0, 'bicgstab');
lb{index} = "bicgstab ilu0";
index = index + 1;

test_unprec( ...
    A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    2, M_smo_ilu0, 'min');
lb{index} = "min res ilu0";
index = index + 1;


% [sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
%     tol_inner, maxit_inner, tol_outer, maxit_outer, 0, M_smo_ilu0, 'bicgstab');
% semilogy(0:length(resvec_outer)-1, resvec_outer, '-o'); hold on;
% lb{index} = "bicgstab unprec";
% index = index + 1;

% =================================================================================================================
% k = 100
v = extractEigs(eigs200, 100);
v = orth(v);

test_mgd(...
    A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    1, M_smo_ilu0, 'min');
lb{index} = "min res, k=100";
index = index + 1;

test_mgd(...
    A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    1, M_smo_ilu0, 'bicgstab');
lb{index} = "bicgstab, k=100";
index = index + 1;

test_mgd(...
    A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    3, M_smo_ilu0, 'min');
lb{index} = "min res k=100, ilu(0) smo";
index = index + 1;

test_mgd(...
    A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    3, M_smo_ilu0, 'bicgstab');
lb{index} = "bicgstab k=100, ilu(0) smo";
index = index + 1;

grid on;
legend(lb);
%% Plot Eigen Values
load("eigs200.mat");

d_200 = sqrt(sort(abs(diag(eigs200{2})), "descend"))/sqrt(max(abs(diag(eigs200{2}))));

semilogy(d_200); grid on;
title("Eigenvalue Magnitude Decay");
xlabel("Magnitude Descend");
ylabel("|Eig Val|")

%%
function [v, d] = getEigs(A, k, tol, maxit) 
    n = size(A, 1);
    t=@(A,b)bicgstab(A, b, 0.003, 1000); 
    % ======= inv(A')*y = inv(A') * inv(A) * b =============
    % [v, d] = eigs(@(x)t(A',t(A,x)), n, k, 'largestimag', ...
    %           'Tolerance',tol,'MaxIterations',maxit);
    
    % ======= inv(A)* y = inv(A) * inv(A)' * b =============
    [v, d] = eigs(@(x)t(A,t(A',x)), n, k, 'largestimag', ...
              'Tolerance',tol,'MaxIterations',maxit);
    eigCell = {v, d};
    save("eigs200_1e1_1000.mat", "eigCell")
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

% geometric_mean prod(resvec_1,....,resvec100)^{1/100}
% ==> exp(mean(log(X))
function test_unprec( ...
    A, B, v, ...
    tol_inner, maxit_inner, ...
    tol_outer, maxit_outer, ...
    precond, M_smo_ilu0, solver)

m = size(B,2);
lens = zeros(m,1);
log_curves = cell(m, 1);

for j = 1:m
    rhs = B(:,j);
    [~, ~, ~, resvec_outer, ~] = MG_deflation( ...
        A, rhs, v, ...
        tol_inner, maxit_inner, ...
        tol_outer, maxit_outer, ...
        precond, M_smo_ilu0, solver);

    log_curves{j} = log10(resvec_outer(:));
    lens(j) = numel(resvec_outer(:));
end

Lmax = max(lens);
resvec_matrix = NaN(Lmax, m); % NaN Matrix Padding
for j = 1:m
    L = lens(j);
    resvec_matrix(1:L, j) = log_curves{j};
end

avg_log = mean(resvec_matrix, 2, 'omitnan'); 
avg_curve = exp(avg_log);     % exp(mean(log))

if strcmpi(solver, 'bicgstab'), mstyle = '-';
else 
    mstyle = '--'; 
end
if precond == 0
    semilogy(avg_curve, 'LineStyle', '-', 'Marker','o'); hold on;
else 
    semilogy(avg_curve, 'LineStyle', mstyle); hold on;
end
end

function test_mgd( ...
    A, B, v, ...
    tol_inner, maxit_inner, ...
    tol_outer, maxit_outer, ...
    precond, M_smo_ilu0, solver)
    if nargin < 10 || isempty(solver) || strcmpi(solver, 'bicgstab')
        mstyle = '-';
    else
        mstyle = '--';
    end

    m = size(B, 2);

    X_list = cell(m, 1);
    Y_list = cell(m, 1);
    Xmax = 0;

    for j = 1:m
        rhs = B(:, j);
        [~, ~, inner_iter_vec, resvec_outer, ~] = MG_deflation( ...
            A, rhs, v, ...
            tol_inner, maxit_inner, ...
            tol_outer, maxit_outer, ...
            precond, M_smo_ilu0, solver);

        y = resvec_outer(2:end);
        x = cumsum(inner_iter_vec(:));
        assert(numel(x) == numel(y), 'inner_iter_vec size must match resvec_outer(2:end)');

        X_list{j} = x(:);

        Y_list{j} = max(y(:), eps);

        if ~isempty(x)
            Xmax = max(Xmax, x(end));
        end
    end

    if Xmax == 0
        warning('No valid samples to average.');
        return;
    end

    % 投到统一的工作量网格：1..Xmax，在各自的 x 位置上放入 log(y)，其它为 NaN
    LogY = NaN(Xmax, m);
    for j = 1:m
        x = X_list{j};
        y = Y_list{j};
        if isempty(x), continue; end
        LogY(x, j) = log(y);
    end

    avg_log = mean(LogY, 2, 'omitnan');
    avg_curve = exp(avg_log);
    X = (1:Xmax).';
    mask = ~isnan(avg_curve);

    semilogy(X(mask), avg_curve(mask), 'LineWidth', 2, 'LineStyle', mstyle); hold on;
    xlabel('Cumulative inner iterations');
    ylabel('residual norm (resvec)');
    % title(sprintf('Average of %d runs (geom. mean on work grid)', m));
end

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
