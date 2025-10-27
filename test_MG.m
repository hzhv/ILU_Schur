function test_MG
%% Noted these tests only for plotting, 200 eigs
% Outer Solver Options: bicgstab, min_res
% inner Solver: GMRES

m = 2; % # of RHSs
tol_inner = 0.1; maxit_inner = 4;
tol_outer = 1e-3; maxit_outer = 15;

A = load('./A_level2.mat').A; % Not Hermitian
n = size(A, 1);               % Hermitian: diag of A are real number
bs = 64; dim=[4 4 4 8];

rhs = load('./rhs_level2.mat').x;
rhs = rhs(:, 1:m);
% rhs = randn(size(A,1),1);

[L, U] = ilu(A, struct('type','nofill'));
M_smo_ilu0 = @(x) U\(L\x);

bj = invblkdiag(A, bs);
M_smo_bj = @(x) bj * x;

eigs200 = load("eigs200.mat").eigCell;
v = extractEigs(eigs200, 64);
% v = v.*(1+0.1*complex(randn(n,100),randn(n,100)));
v = orth(v);

Triplets = load("singularTrip.mat").SCell;
Us = Triplets{1}; Ss = Triplets{2}; Vs = Triplets{3};

r={};
lb = {};
index = 1;


% =================== For Schur ======================================
p = coloring(dim,bs,1,1,zeros(size(dim)));
[~, perm] = sort(p);
Ap = A(perm, perm);  % Colored
for i = 1:m, rhsp(:,i) = rhs(perm,i); end

a00 = A(p==0,p==0);
a01 = A(p==0,p==1);
a10 = A(p==1,p==0);
a11 = A(p==1,p==1);
assert(nnz(blkdiag(a00, bs)-a00) == 0) 
inva11 = invblkdiag(a11,bs);
% s = @(x) a00*x - a01*(inva11*(a10*x));
disp("Explicitly Calculating ilu0(s)...")
s = a00 - a01*(inva11*(a10));
rhs0 = rhs(p==0,:) - a01*(inva11*rhs(p==1,:));

[lSch, uSch] = ilu(s, struct('type','nofill'));
M_Schur_ilu0 = @(x) uSch\(lSch\x);

[Lp, Up] = ilu(Ap, struct('type','nofill'));
M_Aperm_ilu0 = @(x) Up\(Lp\x);

bjs = invblkdiag(s, bs);
assert(mod(size(s,1),bs)==0);
M_Schur_bj = @(x) bjs * x;

SchurTrip = load("SchurSingularTrip.mat").SCell;
USch = SchurTrip{1}; SSch = SchurTrip{2}; VSch = SchurTrip{3};
%%
disp("Tests start...")
% ====================== S S ===============================
r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    0, M_smo_ilu0, 'min');
lb{index} = "MinRes(S), unprec";
index = index + 1;

r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    1, M_smo_ilu0, 'min');
lb{index} = "MinRes(S, defl)";
index = index + 1;

r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    2, M_Schur_ilu0, 'min');
lb{index} = "MinRes(S, ilu0(S))";
index = index + 1;

r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    3, M_Schur_ilu0, 'min');
lb{index} = "MinRes(S, defl(ilu0(S)))";
index = index + 1;

r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    4, M_Schur_bj, 'min');
lb{index} = "MinRes(S, bj(S))";
index = index + 1;

r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    5, M_Schur_bj, 'min');
lb{index} = "MinRes(S, defl(bj(S)))";
index = index + 1;
lineSS = index;
% ====================== S A00 ===============================
p=coloring(dim,bs,1,1,zeros(size(dim)));
[l,u]=ilu0_colors(A,p,bs);

M_Schur_ilu0A = @(x)select_dom(solve_ilu(l,u,p,bs,expand_from_dom(x,p,0)),p,0);
r{index} = test_mgd_singular( ...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    2, M_Schur_ilu0A, 'min');

lb{index} = "MinRes(S), ilu0(A00)";
index = index + 1;
lineSAoo = index;
% ====================== A(2-color) ===============================
r{index} = test_mgd_singular(...
    Ap, rhsp, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    0, M_smo_ilu0, 'min');
lb{index} = "MinRes(A(2-color)), unprec";
index = index + 1;

r{index} = test_mgd_singular(...   
    Ap, rhsp, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    1, M_smo_ilu0, 'min');
lb{index} = "MinRes(A(2-color), defl)";
index = index + 1;

r{index} = test_mgd_singular(...
    Ap, rhsp, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    2,  M_smo_ilu0, 'min');
lb{index} = "MinRes(A(2-color), ilu0(A))";
index = index + 1;

r{index} = test_mgd_singular(...
    Ap, rhsp, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    3, M_smo_ilu0, 'min');
lb{index} = "MinRes(A(2-color), defl(ilu0(A)))";
index = index + 1;

r{index} = test_mgd_singular(...
    Ap, rhsp, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    5,  M_smo_bj, 'min');
lb{index} = "MinRes(A(2-color), defl(bj(A)))";
index = index + 1;

r{index} = test_mgd_singular(...
    Ap, rhsp, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    2,  M_Aperm_ilu0, 'min');
lb{index} = "MinRes(A(2-color), ilu0(A(2-color)))";
index = index + 1; 

r{index} = test_mgd_singular(...
    Ap, rhsp, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    5,  M_Aperm_ilu0, 'min');
lb{index} = "MinRes(A(2-color), defl(ilu0(A(2-color))))";
index = index + 1; 
lineA2Color = index;
% ====================== Regular ===============================
r{index} = test_mgd_singular(...
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    0, M_smo_ilu0, 'min');
lb{index} = "MinRes(A), unprec";
index = index + 1;

r{index} = test_mgd_singular(...   
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    1, M_smo_ilu0, 'min');
lb{index} = "MinRes(A, defl)";
index = index + 1;

r{index} = test_mgd_singular(...
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    2,  M_smo_ilu0, 'min');
lb{index} = "MinRes(A, ilu0(A))";
index = index + 1;

r{index} = test_mgd_singular(...
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    3, M_smo_ilu0, 'min');
lb{index} = "MinRes(A, defl(ilu0(A)))";
index = index + 1;

r{index} = test_mgd_singular(...
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    4,  M_smo_bj, 'min');
lb{index} = "MinRes(A, bj(A))";
index = index + 1;

r{index} = test_mgd_singular(...
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    5,  M_smo_bj, 'min');
lb{index} = "MinRes(A, defl(bj(A)))";
index = index + 1;
line = index;

% PLOT
f = figure;
clf
for i=1:numel(r)
	if i < lineSS
		semilogy(r{i}, '-', 'linewidth',2);
	elseif i < lineSAoo
		semilogy(r{i}, '--', 'linewidth',2);
    elseif i < lineA2Color
        semilogy(r{i}, '-o', 'linewidth',2);
    elseif i < line
		semilogy(r{i}, '-x', 'linewidth',2);
	end
	if i == 1, hold on; end
end
yline(tol_outer ,'r-.','DisplayName', sprintf('Tol'));

grid on;
legend(lb);
% title(sprintf('Average of %d RHS (geom. mean)', m));

savefig(f, 'MG_avg_50.fig');
end

%%
function [v, d] = getEigs(A, k, tol, maxit) 
    n = size(A, 1);
    t=@(A,b)bicgstab(A, b, 0.003, 1000); 
    % No! inv(A')*y = inv(A') * inv(A) * b
    % ======= inv(A)* y = inv(A) * inv(A)' * b =============
    % 
    [v, d] = eigs(@(x) t(A, t(A',x)), n, k, 'largestimag', ...
              'Tolerance',tol,'MaxIterations',maxit);
    eigCell = {v, d};
    save("eigs200_1e1_1000.mat", "eigCell")
end
   
function [v, d] = extractEigs(eigs, k)
    if nargin < 2, k = 1; end
    V = eigs{1};
    D = eigs{2};
    d = diag(D);
    [~, perm] = sort(d, "descend"); 
    permD = D(perm, perm);
    permV = V(:, perm); 
    v = permV(:, 1:k);
    d = permD(1:k, 1:k);
end

function [u, s, v] = extractSTriplets(triplets, k)
    if nargin < 2, k = 100; end
    U = triplets{1};
    S = triplets{2};
    V = triplets{3};
    d = diag(S);
    [~, perm] = sort(d, "descend");
    
    permU = U(:, perm);    u = permU(:, 1:k);
    permS = S(perm, perm); s = permS(1:k, 1:k);
    permV = V(:, perm);    v = permV(:, 1:k);
end

function geo_mean = test_mgd_singular(...
    A, B, u, v, ...
    tol_inner, maxit_inner, ...
    tol_outer, maxit_outer, ...
    precond, M_smo, solver)

m = size(B, 2);
Xmax_each = zeros(m,1);         % gather total inner iteratons per rhs
interp_ln_resvecs = cell(m,1);  % interp on 1:Xmax, gather'em all

for j = 1:m
        rhs = B(:, j);
        [~, ~, inner_iter_vec, resvec_outer, ~] = MG_deflation_Singular( ...
            A, rhs, u, v, ...
            tol_inner, maxit_inner, ...
            tol_outer, maxit_outer, ...
            precond, M_smo, solver);
    resvec_outer = resvec_outer/norm(rhs);

    if precond == 0 || precond == 2 || precond == 4
        X = (1:numel(resvec_outer)); 
    else
        X = [1; cumsum(inner_iter_vec(:))];
    end
    assert(numel(X) == numel(resvec_outer), ...
        'inner_iter_vec size must match resvec_outer(2:end)');
    [Xu, ia] = unique(X, 'stable');  % X_unique
    resvec_outer = resvec_outer(ia);

    Xq = (1:max(Xu));
    ln_resvec_outer = log(resvec_outer);
    ln_resvec_q = interp1(Xu, ln_resvec_outer, Xq, 'linear');
    interp_ln_resvecs{j} = ln_resvec_q;
    Xmax_each(j) = max(Xu);
end
% Truncation
Lmin = floor(min(Xmax_each));
resvec_matrix = NaN(Lmin, m); % construct convergence history for all rhs
for j = 1:m
    resvec_matrix(:, j) = interp_ln_resvecs{j}(1:Lmin);
end
if anynan(resvec_matrix)
    fprintf("\n%s solover, precond=%g\n", solver, precond);
    error("NaN in conv history");
end
    
mean_log = mean(resvec_matrix, 2); %'omitnan'); 
geo_mean = exp(mean_log);
% 
% if size(A,1) < 30000 % TODO...
%    mstyle = '--'; 
% else 
%    mstyle = '-';
% end
% % if colorTest == 1, mstyle = '-'; end
% 
% if precond == 0
%     semilogy(geo_mean, 'linewidth',2, 'Color', "r", 'LineStyle', mstyle); hold on;
% elseif precond == 1
%     semilogy(geo_mean, 'linewidth',2, 'LineStyle', mstyle, 'Marker','o'); hold on;
% elseif precond == 2 || precond == 3
%     semilogy(geo_mean, 'linewidth',2, 'LineStyle', mstyle, 'Marker','^'); hold on;
% elseif precond == 4 || precond == 5
%     semilogy(geo_mean, 'linewidth',2, 'LineStyle', mstyle, 'Marker','+'); hold on;
% end

end

function test_mgd(...
    A, B, v, ...
    tol_inner, maxit_inner, ...
    tol_outer, maxit_outer, ...
    precond, M_smo, solver)

m = size(B, 2);
Xmax_each = zeros(m,1);         % gather total inner iteratons per rhs
interp_ln_resvecs = cell(m,1);  % interp on 1:Xmax, gather'em all

for j = 1:m
        rhs = B(:, j);
        [~, ~, inner_iter_vec, resvec_outer, ~] = MG_deflation( ...
            A, rhs, v, ...
            tol_inner, maxit_inner, ...
            tol_outer, maxit_outer, ...
            precond, M_smo, solver);
    resvec_outer = resvec_outer/norm(rhs);

    if precond == 0 || precond == 2 || precond == 4
        X = (1:numel(resvec_outer)); 
    else
        X = [1; cumsum(inner_iter_vec(:))];
    end
    assert(numel(X) == numel(resvec_outer), ...
        'inner_iter_vec size must match resvec_outer(2:end)');
    [Xu, ia] = unique(X, 'stable');  % X_unique
    resvec_outer = resvec_outer(ia);

    Xq = (1:max(Xu));
    ln_resvec_outer = log(resvec_outer);
    ln_resvec_q = interp1(Xu, ln_resvec_outer, Xq, 'linear');
    interp_ln_resvecs{j} = ln_resvec_q;
    Xmax_each(j) = max(Xu);
end
% Truncation
Lmin = floor(min(Xmax_each));
resvec_matrix = NaN(Lmin, m); % construct convergence history for all rhs
for j = 1:m
    resvec_matrix(:, j) = interp_ln_resvecs{j}(1:Lmin);
end
if anynan(resvec_matrix)
    fprintf("\n%s solover, precond=%g\n", solver, precond);
    error("NaN in conv history");
end
    
mean_log = mean(resvec_matrix, 2); %'omitnan'); 
geo_mean = exp(mean_log);
% Plot
% if strcmpi(solver, 'bicgstab'), mstyle = '-';
if size(A,1) < 30000 % TODO...
   mstyle = '--'; 
else 
   mstyle = '-';
end

if precond == 0
    semilogy(geo_mean, 'linewidth',1.5, 'LineStyle', mstyle, 'Marker','o'); hold on;
elseif precond == 1
    semilogy(geo_mean, 'linewidth',1.5, 'LineStyle', mstyle, 'Marker','.'); hold on;
elseif precond == 2 || precond == 3
    semilogy(geo_mean, 'linewidth',1.5, 'LineStyle', mstyle); hold on;
elseif precond == 4 || precond == 5
    semilogy(geo_mean, 'linewidth',1.5, 'LineStyle', mstyle, 'Marker','x'); hold on;
end

end


function y = expand_from_dom(x, p, c)
	y = zeros(numel(p),size(x,2));
	y(p == c,:) = x;
end

function y = select_dom(x, p, c)
	y = x(p==c,:);
end

% [sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, ...
%     tol_inner, maxit_inner, tol_outer, maxit_outer, ...
%     1, M_smo_ilu0, 'min');
% plotMGD(inner_iter_vec,resvec_outer, 'min'); hold on;
% lb{index} = "min res, k=100";
% index = index + 1;