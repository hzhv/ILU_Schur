function test_MG(m)
% Noted these tests only for plotting

A = load('./A_level2.mat').A;
n = size(A,1);
rhs = load('./rhs_level2.mat').x;

rhs = rhs(:,1:m); %% try more rhs plz

setup.type    = 'nofill';
setup.droptol = 0;  
[L, U] = ilu(A, setup);
M_smo_ilu0 = @(x) U\(L\x);

tol_inner = 0.01; maxit_inner = 4;
tol_outer = 1e-2; maxit_outer = 30;

lb = {};
index = 1;
load("eigs200.mat");

v = extractEigs(eigs200, 100);
% v = v.*(1+0.1*complex(randn(n,100),randn(n,100)));
v = orth(v);

f = figure;
test_mgd( ...
    A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    0, M_smo_ilu0, 'bicgstab');
lb{index} = "bicgstab unprec";
index = index + 1;

test_mgd( ...
    A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    0, M_smo_ilu0, 'min');
lb{index} = "min res unprec";
index = index + 1;

test_mgd( ...
    A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    2, M_smo_ilu0, 'bicgstab');
lb{index} = "bicgstab ilu0";
index = index + 1;

test_mgd( ...
    A, rhs, v, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    2, M_smo_ilu0, 'min');
lb{index} = "min res ilu0";
index = index + 1;

% [sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
%     tol_inner, maxit_inner, tol_outer, maxit_outer, ...
%     1, M_smo_ilu0, 'min');
% plotMGD(inner_iter_vec,resvec_outer, 'min'); hold on;
% lb{index} = "min res, k=100";
% index = index + 1;

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

yline(tol_outer,'r-.','DisplayName', sprintf('Tol'));
grid on;
legend(lb);
xlabel('Cumulative inner iterations');
ylabel('residual norm (resvec)');
title(sprintf('Average of %d RHS (geom. mean)', m));

savefig(f, 'MG_test_avg.fig');
saveas(f, 'MG_test_avg.pdf');

end
%% Plot Eig Values
function plot_eigValues
load("eigs200.mat");

d_200 = sqrt(sort(abs(diag(eigs200{2})), "descend"))/sqrt(max(abs(diag(eigs200{2}))));

semilogy(d_200); grid on;
title("Eigenvalue Magnitude Decay");
xlabel("Magnitude Descend");
ylabel("|Eig Val|")
end
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

    log_curves{j} = log(resvec_outer(:));
    lens(j) = numel(resvec_outer(:));
end

Lmin = min(lens);
resvec_matrix = NaN(Lmin, m); % NaN Matrix Padding
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

function test_mgd(A, B, v, ...
    tol_inner, maxit_inner, ...
    tol_outer, maxit_outer, ...
    precond, M_smo_ilu0, solver)

m = size(B, 2);
Xmax_each = zeros(m,1);         % gather total inner iteratons per rhs
interp_ln_resvecs = cell(m,1);  % interp on 1:Xmax, gather'em all

for j = 1:m
        rhs = B(:, j);
        [~, ~, inner_iter_vec, resvec_outer, ~] = MG_deflation( ...
            A, rhs, v, ...
            tol_inner, maxit_inner, ...
            tol_outer, maxit_outer, ...
            precond, M_smo_ilu0, solver);

    resvec_outer = resvec_outer(2:end);
    if precond == 0 || precond == 2
        X = (1:numel(resvec_outer)); 
    else
        X = cumsum(inner_iter_vec);
    end
    
    assert(numel(X) == numel(resvec_outer), 'inner_iter_vec size must match resvec_outer(2:end)');
    [Xu, ia] = unique(X, 'stable');  % X_unique
    resvec_outer = resvec_outer(ia);

    Xq = (1:max(Xu));
    ln_resvec_outer = log(resvec_outer);
    ln_resvec_q = interp1(Xu, ln_resvec_outer, Xq, 'linear');
    interp_ln_resvecs{j} = ln_resvec_q;
    Xmax_each(j) = max(Xu);
end
% Truncation
Lmin = min(Xmax_each(Xmax_each>0));
resvec_matrix = NaN(Lmin, m); % NaN padding
for j = 1:m
    resvec_matrix(:, j) = interp_ln_resvecs{j}(1:Lmin);
end
mean_log = mean(resvec_matrix, 2, 'omitnan'); 
                                     
geo_mean = exp(mean_log);
% Plot
if strcmpi(solver, 'bicgstab'), mstyle = '-';
else 
    mstyle = '--'; 
end

if precond == 0
    semilogy(geo_mean, 'LineStyle', '-', 'Marker','o'); hold on;
else 
    semilogy(geo_mean, 'LineStyle', mstyle); hold on;
end

end

function plotMGD(inner_iter_vec, resvec_outer, solver)
    if nargin < 3 || isempty(solver) || strcmpi(solver, 'bicgstab')
        m = "-";
    else
        m = "--";
    end
    resvec_outer = resvec_outer(2:end);
    assert(size(inner_iter_vec,1) == size(resvec_outer,1), 'inner_iter_vec size must match resvec_outer(2:end)')
    X = cumsum(inner_iter_vec);
    semilogy(X, resvec_outer, LineStyle=m);
end
