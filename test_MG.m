% function test_MG
%% Noted these tests only for plotting, 200 eigs
% Outer Solver Options: bicgstab, min_res
% inner Solver: GMRES

m = 100; % # of RHSs
tol_inner = 0.1; maxit_inner = 4;
tol_outer = 1e-3; maxit_outer = 15;

A = load('./A_level2.mat').A; % Not Hermitian
n = size(A, 1);               % Hermitian: diag of A are real number
bs = 64; dim=[4 4 4 8];

rhs = load('./rhs_level2.mat').x;
rhs = rhs(:, 1:m);
% rhs = randn(size(A,1),1);

eigs200 = load("eigs200.mat").eigCell;
v = extractEigs(eigs200, 64);
% v = v.*(1+0.1*complex(randn(n,100),randn(n,100)));
v = orth(v);

Triplets = load("singularTrip.mat").SCell;
Us = Triplets{1}; Ss = Triplets{2}; Vs = Triplets{3};

p = coloring(dim,bs,1,1,zeros(size(dim)));
[~, perm] = sort(p);
Ap = A(perm, perm);  % Colored
for i = 1:m, rhsp(:,i) = rhs(perm,i); end
disp("Explicitly Calculating ilu0(2-color A)...")
[Lp, Up] = ilu(Ap, struct('type','nofill'));
M_Aperm_ilu0 = @(x) Up\(Lp\x);

[L, U] = ilu(A, struct('type','nofill'));
M_smo_ilu0 = @(x) U\(L\x);
 
bj = invblkdiag(A, bs);
M_smo_bj = @(x) bj * x;

% ====================== Schur ===============================
SchurTrip = load("SchurSingularTrip.mat").SCell;
USch = SchurTrip{1}; SSch = SchurTrip{2}; VSch = SchurTrip{3};

a00 = A(p==0,p==0);
a01 = A(p==0,p==1);
a10 = A(p==1,p==0);
a11 = A(p==1,p==1);
assert(nnz(blkdiag(a00, bs)-a00) == 0) 
inva11 = invblkdiag(a11,bs);
% s = @(x) a00*x - a01*(inva11*(a10*x));
s = a00 - a01*(inva11*(a10));
rhs0 = rhs(p==0,:) - a01*(inva11*rhs(p==1,:));

disp("Explicitly Calculating ilu0(s)...")
[lSch, uSch] = ilu(s, struct('type','nofill'));
M_Schur_ilu0 = @(x) uSch\(lSch\x);

bjs = invblkdiag(s, bs);
assert(mod(size(s,1),bs)==0);
M_Schur_bj = @(x) bjs * x;

% ====================== UB-FSAI ===============================
[ubf_A, FL, FU, D]   = unsymBlockFSAI(A, size(A,1)/bs);
[ubf_S, FLD, FUD, ~] = unsymBlockFSAI(s, size(s,1)/bs);


r={};
lb = {};
index = 1;
%%
disp("Tests start...")
% Unprec
r{index} = test_mgd_singular(...
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    0, M_smo_ilu0, 'min');
lb{index} = "MinRes(A), unprec";
index = index + 1;

r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    0, M_Schur_ilu0, 'min');
lb{index} = "MinRes(S), unprec";
index = index + 1;

r{index} = test_mgd_singular(...
    Ap, rhsp, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    0, M_smo_ilu0, 'min');
lb{index} = "MinRes(A(2-color)), unprec";
lineUnprec = index;
index = index + 1;
%%
% defl
% ======================== DD ===============================
domA_idx = partitioning(dim, bs, [1 1 2 2]);

mask_even = (p==0);             % bs*prod(dims)
domS_idx = domA_idx(mask_even); 
dd = domdiag(s, domS_idx);
function z = dd_inv(dd, v, tol, maxit)
    [z, flag, ~, iters] = minres(dd, v, tol, maxit);
    if flag == 0,fprintf('Solving dd*z = v with %g iters\n', iters); end
end
M_dd = @(v) dd_inv(dd, v, 0.1, 5); 


r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    5, M_dd, 'min');
lb{index} = "minRes(S, defl(DD(S, [1 1 2 4])))";
lineDD = index;
index = index + 1;
%%
%
r{index} = test_mgd_singular(...   
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    1, @(x) U\(L\x), 'min');
lb{index} = "MinRes(A, defl)";
index = index + 1;

r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    1, M_Schur_ilu0, 'min');
lb{index} = "MinRes(S, defl)";
lineDEFL1 = index;
index = index + 1;

r{index} = test_mgd_singular(...
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    3, @(x) U\(L\x), 'min');
lb{index} = "MinRes(A, defl(ilu0(A)))";
index = index + 1;

r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    3, M_Schur_ilu0, 'min');
lb{index} = "MinRes(S, defl(ilu0(S)))";
lineDEFL2 = index;
index = index + 1;

r{index} = test_mgd_singular(...
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    5,  M_smo_bj, 'min');
lb{index} = "MinRes(A, defl(bj(A)))";
index = index + 1;

r{index} = test_mgd_singular(...
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    5,  ubf_A, 'minres');
lb{index} = "minres(A, defl(ubf(A)))";
index = index + 1;

r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    5, M_Schur_bj, 'min');
lb{index} = "MinRes(S, defl(bj(S)))";
lineDEFL3 = index;
index = index + 1;

r{index} = test_mgd_singular(...   
    Ap, rhsp, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    1, @(x) U\(L\x), 'min');
lb{index} = "MinRes(A(2-color), defl)";
index = index + 1;

r{index} = test_mgd_singular(...
    Ap, rhsp, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    5,  M_Aperm_ilu0, 'min');
lb{index} = "MinRes(A(2-color), defl(ilu0(A(2-color))))";
index = lineDEFL4;
index = index + 1; 


% No Defl
% ilu(0)
r{index} = test_mgd_singular(...
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    2,  @(x) U\(L\x), 'min');
lb{index} = "MinRes(A, ilu0(A))";
index = index + 1;

r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    2, M_Schur_ilu0, 'min');
lb{index} = "MinRes(S, ilu0(S))";
index = index + 1;

r{index} = test_mgd_singular(...
    Ap, rhsp, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    2,  M_Aperm_ilu0, 'min');
lb{index} = "MinRes(A(2-color), ilu0(A(2-color)))";
index = index + 1; 


[l,u]=ilu0_colors(A,p,bs); % s, ilu0(A00)
M_Schur_ilu0A = @(x)select_dom(solve_ilu(l,u,p,bs,expand_from_dom(x,p,0)),p,0);
r{index} = test_mgd_singular( ...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    2, M_Schur_ilu0A, 'min');
lb{index} = "MinRes(S, ilu0(A00)";
index = index + 1;
%%
% UBFSAI
r{index} = test_mgd_singular(...
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    4, ubf_A, 'minres');
lb{index} = "minres(A, ubf(A)";
index = index + 1;

r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    4, ubf_S, 'min');
lb{index} = "MinRes(S, ubf(S))";
index = index + 1;

r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    5, ubf_S, 'min');
lb{index} = "MinRes(S, defl(ubf(S)))";
index = index + 1;

%%
% DD
r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    4, M_dd, 'min');
lb{index} = "minRes(S, DD(S, [1 1 2 4]))";
index = index + 1;

% bj
r{index} = test_mgd_singular(...
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    4,  M_smo_bj, 'min');
lb{index} = "MinRes(A, bj(A))";
index = index + 1;

r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    4, M_Schur_bj, 'min');
lb{index} = "MinRes(S, bj(S))";

lineBJ = index;
index = index + 1;

%%
% PLOT
f = figure;
clf
plotAutoStyle(r(1:lineUnprec),             'r'); 
plotAutoStyle(r(lineUnprec+1:lineDEFL4),   'g');
plotAutoStyle(r(lineDEFL4+1:lineBJ),       'b'); 
% plotAutoStyle(r(lineILU2+1:lineILU3),   'c'); 
% plotAutoStyle(r(lineILU3+1:lineDEFL1),  'm');
% plotAutoStyle(r(lineDEFL1+1:lineDEFL2), [1 0.5 0]);     % orange
% plotAutoStyle(r(lineDEFL2+1:lineDEFL3), 'k');
% plotAutoStyle(r(lineDEFL3+1:lineDEFL4), [0.6 0.3 0.8]); % purple
% plotAutoStyle(r(lineDEFL4+1:lineBJ),    [1 0.8431 0]);  % yellow
% plotAutoStyle(r(lineBJ+1:lineDD),       [0.243, 0.588, 0.318]);  % Dark Green

yline(tol_outer ,'r-.','DisplayName', sprintf('Tol')); 

grid on;
legend(lb);
ylabel("relative residual norm");
xlabel("Iterations");
% xlabel("S takes 1 sync, ilu(S) takes 3 sync, ilu(perm(A)) 2 colors takes 2 syncs");
% savefig(f, 'MG_avg_50.fig');
% end
%% For Sync
figure(2);

semilogy(1+(0:length(r{2})-1)*1, r{2}, 'LineWidth', 2); hold on;
semilogy(1+(0:length(r{5})-1)*(1+3), r{5}, 'LineWidth', 2); 
semilogy(1+(0:length(r{9})-1)*1, r{9}, 'LineWidth', 2); 
semilogy(1+(0:length(r{11})-1)*(1+3), r{11}, 'LineWidth', 2); 
semilogy(1+(0:length(r{13})-1)*(1+1), r{13}, 'LineWidth', 2); 
semilogy(1+(0:length(r{17})-1)*(1+1), r{17}, 'LineWidth', 2); 
semilogy(1+(0:length(r{18})-1)*(1+3), r{18}, 'LineWidth', 2); 
semilogy(1+(0:length(r{19})-1)*(1+3), r{19}, 'LineWidth', 2); 

grid on;
legend(lb{2}, lb{5}, lb{9}, lb{11}, lb{13} ,lb{17}, lb{18}, lb{19});
%%
function plotAutoStyle(Y, color, sync_factor)
    if nargin < 3, sync_factor = 1; end;

    markers = {'o','x','<','d','v','>','^','p','h'};
    linestyles = {'-','--',':','-.'};
    for i = 1:numel(Y)
        m = markers{mod(i-1, numel(markers))+1};
        % l = linestyles{mod(i-1, numel(linestyles))+1};
        % if numel(Y) ~= 1
            semilogy(Y{i}, ...
                 'Color', color, ...
                 'Marker', m, ...
                 'LineStyle', '-', ...
                 'LineWidth', 2);
            hold on;
        % else
        %     semilogy((1:length(Y))*sync_factor, Y, ...
        %          'Color', color, ...
        %          'Marker', m, ...
        %          'LineStyle', '-', ...
        %          'LineWidth', 2)
        % end
    end
end

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
end


function y = expand_from_dom(x, p, c)
	y = zeros(numel(p),size(x,2));
	y(p == c,:) = x;
end

function y = select_dom(x, p, c)
	y = x(p==c,:);
end
