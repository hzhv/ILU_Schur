function test_L1
% no FSAI tests here for huge L1...
% Outer Solver Options: bicgstab, min_res
% inner Solver: GMRES

m = 1; % # of random RHSs
tol_inner = 0.1; maxit_inner = 4;
tol_outer = 1e-3; maxit_outer = 15;

A = load('./A_l1.mat').A; % Not Hermitian

n = size(A, 1);   
bs = 48; dim=[8 8 8 16];

randn('state',3);
rhs = randn(size(A,1),m);

Triplets = load("singularTripL1_DD_Approx.mat").SCell;
Us = Triplets{1}; Ss = Triplets{2}; Vs = Triplets{3};

p = coloring(dim,bs,1,1,zeros(size(dim)));
[~, perm] = sort(p);
Ap = A(perm, perm);  % Colored
for i = 1:m, rhsp(:,i) = rhs(perm,i); end

disp("Explicitly Calculating ilu0(2-color A)...")
[Lp, Up] = ilu(Ap, struct('type','nofill'));
disp("Done.");
M_Aperm_ilu0 = @(x) Up\(Lp\x);

disp("Explicitly Calculating ilu0(A)...");
tic;
[L, U] = ilu(A, struct('type','nofill'));
M_smo_ilu0 = @(x) U\(L\x);
disp("Done."); toc

% bj = invblkdiag(A, bs);
% M_smo_bj = @(x) bj * x;

% ====================== Schur ===============================
SchurTrip = load("singularTripL1_Schur_DD_Approx.mat").SCell;
USch = SchurTrip{1}; SSch = SchurTrip{2}; VSch = SchurTrip{3};

a00 = A(p==0,p==0);
a01 = A(p==0,p==1);
a10 = A(p==1,p==0);
a11 = A(p==1,p==1);
assert(nnz(blkdiag(a00, bs)-a00) == 0) 
inva11 = invblkdiag(a11,bs);
% % s = @(x) a00*x - a01*(inva11*(a10*x));
s = a00 - a01*(inva11*(a10));
rhs0 = rhs(p==0,:) - a01*(inva11*rhs(p==1,:));

disp("Explicitly Calculating ilu0(s)..."); 
tic;
[lSch, uSch] = ilu(s, struct('type','nofill'));
M_Schur_ilu0 = @(x) uSch\(lSch\x);
disp("Done."); toc

% bjs = invblkdiag(s, bs);
% assert(mod(size(s,1),bs)==0);
% M_Schur_bj = @(x) bjs * x;

%% ======================== DD ===============================
tic; disp("Preparing Domain Decompostion preconditioner...");
domA_idx = partitioning(dim, bs, [1 1 2 4]);

mask_even = (p==0);             % bs*prod(dims)
domS_idx = domA_idx(mask_even); 
dd = domdiag(s, domS_idx);

M_dd = @(v) dd_inv(dd, v, 0.1, 5);
disp("Done"); toc
% ====================== UB-FSAI ===============================
% disp("Preparing FSAI...");
% [ubf_A, FL, FU, D]    = unsymBlockFSAI(A, size(A,1)/bs);
% [ubf_S, FLs, FUs, Ds] = unsymBlockFSAI(s, size(s,1)/bs);
% disp("Done.");

% [~, perm_dom]  = sort(domA_idx);
% [~, perm_doms] = sort(domS_idx);
% A_dd = A(perm_dom, perm_dom); s_dd = s(perm_doms, perm_doms);
% rhs_dd = rhs(perm_dom, :); 
% rhs0_dd = rhs0(perm_doms, :);
% 
% disp("Preparing Denser FSAI for A...");
% [ubf_denseA, FL_denseA, FU_denseA, ~] = unsymBlockFSAI(A_dd, prod([1 1 2 4]));
% 
% disp("Preparing Denser FSAI for s...");
% [ubf_denseS, FL_denseS, FU_denseS, ~] = unsymBlockFSAI(s_dd, prod([1 1 2 4]));
% disp("Done.");
% =============================================================
r={};
lb = {};
index = 1;


disp("Tests start...")
% ====================== Unprec =============================== 
% r{index} = test_mgd_singular(...
%     Ap, rhsp, Us, Vs, ...
%     tol_inner, maxit_inner, tol_outer, maxit_outer, ...
%     0, M_smo_ilu0, 'min');
% lb{index} = "MinRes(A(2-color)), unprec";
% lineUnprec = index;
% index = index + 1;
% ===========================================================
%% S
r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    0, M_Schur_ilu0, 'min');
lb{index} = "MinRes(S)";
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
    4, M_dd, 'min');
lb{index} = "minRes(S, DD(S, [1 1 2 4]))";
lineS = index;
index = index + 1;

% r{index} = test_mgd_singular(...
%     s, rhs0, USch, VSch, ...
%     tol_inner, maxit_inner, tol_outer, maxit_outer, ...
%     4, ubf_S, 'min');
% lb{index} = "MinRes(S, ubf(S))";
% index = index + 1;

% r{index} = test_mgd_singular(...
%     s_dd, rhs0_dd, USch, VSch, ...
%     tol_inner, maxit_inner, tol_outer, maxit_outer, ...
%     2, ubf_denseS, 'min');
% lb{index} = "minRes(S, ubf(S, [1 1 2 4]))";
% lineS = index;
% index = index + 1;
% -------------------------------------------------------------------------
r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    1, M_Schur_ilu0, 'min');
lb{index} = "MinRes(S, defl)";
index = index + 1;

r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    3, M_Schur_ilu0, 'min');
lb{index} = "MinRes(S, defl(ilu0(S)))";
index = index + 1;

% r{index} = test_mgd_singular(...
%     s, rhs0, USch, VSch, ...
%     tol_inner, maxit_inner, tol_outer, maxit_outer, ...
%     5, ubf_S, 'min');
% lb{index} = "MinRes(S, defl(ubf(S)))";
% lineUBF = index;
% index = index + 1;

r{index} = test_mgd_singular(...
    s, rhs0, USch, VSch, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    5, M_dd, 'min');
lb{index} = "minRes(S, defl(DD(S, [1 1 2 4])))";
lineSdefl = index;
index = index + 1;

% r{index} = test_mgd_singular(...
%     s_dd, rhs0_dd, USch, VSch, ...
%     tol_inner, maxit_inner, tol_outer, maxit_outer, ...
%     5, ubf_denseS, 'min');
% lb{index} = "minRes(S, defl(ubf(S, [1 1 2 4])))";
% lineSdefl = index;
% index = index + 1;

% ============= A ==================================================
r{index} = test_mgd_singular(...
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    0, M_smo_ilu0, 'min');
lb{index} = "MinRes(A)";
index = index + 1;

r{index} = test_mgd_singular(...
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    2,  @(x) U\(L\x), 'min');
lb{index} = "MinRes(A, ilu0(A))";
index = index + 1;

r{index} = test_mgd_singular(...
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    4, M_ddA, 'min');
lb{index} = "minRes(A, DD(S, [1 1 2 4]))";
lineA = index;
index = index + 1;

% r{index} = test_mgd_singular(...
%     A, rhs, Us, Vs, ...
%     tol_inner, maxit_inner, tol_outer, maxit_outer, ...
%     4, ubf_A, 'minres');
% lb{index} = "minres(A, ubf(A))";
% index = index + 1;

% r{index} = test_mgd_singular(...
%     A_dd, rhs_dd, Us, Vs, ...
%     tol_inner, maxit_inner, tol_outer, maxit_outer, ...
%     2, ubf_denseA, 'min');
% lb{index} = "minRes(A, ubf(A, [1 1 2 4]))";
% lineA = index;
% index = index + 1;
% -----------------------------------------------------------------
r{index} = test_mgd_singular(...   
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    1, @(x) U\(L\x), 'min');
lb{index} = "MinRes(A, defl)";
index = index + 1;

r{index} = test_mgd_singular(...
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    3, @(x) U\(L\x), 'min');
lb{index} = "MinRes(A, defl(ilu0(A)))";
index = index + 1;

r{index} = test_mgd_singular(...
    A, rhs, Us, Vs, ...
    tol_inner, maxit_inner, tol_outer, maxit_outer, ...
    5, M_ddA, 'min');
lb{index} = "minRes(A, defl(DD(A, [1 1 2 4])))";
lineAdefl = index;
index = index + 1;

% r{index} = test_mgd_singular(...
%     A, rhs, Us, Vs, ...
%     tol_inner, maxit_inner, tol_outer, maxit_outer, ...
%     5,  ubf_A, 'minres');
% lb{index} = "MinRes(A, defl(ubf(A)))";
% index = index + 1;

% r{index} = test_mgd_singular(...
%     A_dd, rhs_dd, Us, Vs, ...
%     tol_inner, maxit_inner, tol_outer, maxit_outer, ...
%     5, ubf_denseA, 'min');
% lb{index} = "minRes(A, defl(ubf(A, [1 1 2 4])))";
% lineAdefl = index;
% index = index + 1;


save('results_L1.mat', 'r');

%% PLOT
f = figure;
clf
plotAutoStyle(r(1:lineS),           2.5, '-.');
plotAutoStyle(r(lineS+1:lineSdefl), 2.5, '-.', 'o');
plotAutoStyle(r(lineSdefl+1:lineA), 2.5, '-');
plotAutoStyle(r(lineA+1:lineAdefl), 2.5, '-',  'x');

yline(tol_outer ,'r-.','DisplayName', sprintf('Tol')); 

grid on;
legend(lb);
ylabel("relative residual norm");
xlabel("iterations");
savefig(f, 'l1_iters.fig');
end

function plotAutoStyle(Y, lineWidth, linestyle, marker)
    if nargin < 4, marker = "none"; end
    colors = {'m', 'r', 'g', 'b', 'c'};
    % markers = {'<','d','v','>','^','p','h'};
    for i = 1:numel(Y)
        c = colors{mod(i-1, numel(colors))+1};
        semilogy(Y{i}, ...
             'Color', c, ...
             'Marker', marker, ...
             'LineStyle', linestyle, ...
             'LineWidth', lineWidth, ...
             'MarkerSize', 5);
        hold on;
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
