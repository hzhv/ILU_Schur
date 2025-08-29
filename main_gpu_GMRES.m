% 08/01/2025
% Problems:
%  Preconditioners of GPU_GMRES in the Schur Complement section
%  1. try Bicgstab if GMRES failed
%  2. try [8 8 8 16], bs = 48; RHS=[1:N]; p = [0 0 0 2], all 0, all 1, [1 -1 1 -1]
%% Testing the followings on GPU:
%
%   Permute matrix with multicoloring, even-odd reordering, 
%   apply ILU(0) preconditioner, 
%   the last part for implementing the Schur complement for solving Ax = b.
%
%  Main args:
%    D       : length N vector for generating Nd matrix
%    p       : length N vector for displacement at each d
%    k_total : 1:k_distance, i.e. A^k
%   
%    A       : NxN Sparse matrix (square, nonsingular)
%    Colors  : Nx1 permutation matrix 
%    b       : Right-hand side vector
%    tol     : tolerance for iterative solves

clc; clear; reset(gpuDevice);
% close all;
rng(1); parallel.gpu.rng(1, 'Philox');

% === Hyper Params ===
maxit = 1000;
restart = 10;
p=[0 0 0 0];
k_total=3;
tol=1e-2; % coarse operator

% === Sparse System ===
D = [4 4 4 8];            % from read_coarse.m  
A = load('A.mat').A; 
bs = 64;                  % block size of each "block" in diagonal

% D = [8 8 8 16];
% A = lap_kD_periodic(D,1);  
% A = kron(A, ones(bs));    % For denser "boxed" diag
% A = A + speye(size(A,1))*1e-2;  % make it non-singular
% bs = 48;

B_perm = ones(bs,1);        % for Kron(A, B)
N_new = size(A,1);
Ag = gpuArray(A);

%%A = A.*(1+rand(N,N)/10) + speye(N)*1e-7;
% figure; spy(A); title("Original A");

% ======= RHS ========
b = load('rhs.mat').b;
% b = (1:N_new)';
% b = rand(N_new,10);
bg = gpuArray(b);

% x0 = b;
%% "Pure GMRES Iterations"
tic;
fprintf('Pure iterative results w/o preconditioner:\n');
[xg, flag, relres, iter, resvec_GMRES] = ... % 
    gmres(Ag, bg, restart, tol, maxit, [], []);
tp = toc;

relres_true = norm(bg - Ag*xg)/norm(bg);
total_iter_pure = (iter(1)-1)*restart + iter(2);
fprintf('  The actual residual norm = %d\n', relres_true);
fprintf('  GMRES projection relative residual %e in %d iterations.\n', relres, total_iter_pure);
fprintf('  GMRES cost %d sec\n\n', tp);
%% Use ILU(0) Preconditioner Only
fprintf('Use ilu(0) but no multi-reordering:\n');
setup.type    = 'nofill';
setup.droptol = 0;  
[L, U] = ilu(A, setup); % Matlab can't do ILU on GPU
Lg = gpuArray(L); Ug = gpuArray(U);

M_handle = @(x) Ug\(Lg\x);

tic;
for bdx = 1:size(bg,2)
[x_perm_gpu, flag, relres, iter, resvec_ilu0] = ...
    gmres(Ag, bg(:, bdx), restart, tol, maxit, [], M_handle); % Handle
t_ilu = toc;

relres_true_ilu0 = norm(bg - Ag*x_perm_gpu)/norm(bg);
total_iter_ilu0 = (iter(1)-1)*restart + iter(2);
fprintf('  The actual residual norm = %d\n', relres_true_ilu0);
fprintf('  GMRES projection relative residual %e in %d iterations.\n', relres, total_iter_ilu0);
fprintf('  GMRES cost %d sec\n\n', t_ilu);
end
%% Use muticoloring only without Even-Odd ordering
fprintf('Only multi-coloring w/o Even-Odd:\n');
evenCs_noEO = zeros(1, k_total);
for k = 1:k_total
    [Colors, ncolor] = displacement_even_odd_coloring_nD_lattice(D, k, [0 0 0 0]);
    [isOK, evenCs, badCs, loc] = check_eo_compatibility_and_return_even(Colors, D, N_new);
    evenCs_noEO(k) = length(unique(evenCs));
    fprintf('For k = %g, the number of colors = %g:\n', k, ncolor);
    if ~isOK
        fprintf('  Not compatible. Conflicting colors: '); fprintf('%d ', badCs); fprintf('\n');
        disp(loc{1});
        error('Aborted.')
    else
        fprintf('  OK: coloring is even/odd compatible.\n');
    end
end

iters = zeros(1, k_total);
res_noEO = cell(1, k_total);
nColors_noEO = zeros(1, k_total);
for k = 1:k_total
    [Colors, nColors] = displacement_coloring_nD_lattice(D, k, p);
    Colors = kron(Colors, B_perm);
    [~, perm] = sort(Colors);
    A_perm = A(perm, perm);
    A_perm_gpu = gpuArray(A_perm);  % Copy to GPU
    b_perm_gpu = bg(perm);
    
    % CPU ILU then copy to GPU
    setup.type    = 'nofill';
    setup.droptol = 0;  
    [L, U] = ilu(A_perm, setup); % Matlab can't do ILU on GPU
    Lg = gpuArray(L); Ug = gpuArray(U);
    M_handle = @(x) Ug\(Lg\x);

    % x0 = b_perm;
    tic;
    [x_perm_gpu, flag, relres, iter, resvec_noEO] = ...
        gmres(A_perm_gpu, b_perm_gpu, restart, tol, maxit, [], M_handle);
    t_noEO = toc;
    
    total_iter = (iter(1)-1)*restart + iter(2);
    relres_true_ = norm(b_perm_gpu - A_perm_gpu*x_perm_gpu)/norm(b_perm_gpu);
    
    if flag == 0
        fprintf('  When k = %d,', k);
        fprintf('  total colors = %d\n', nColors);
        fprintf('  the actual residual norm = %d\n', relres_true_);
        fprintf('  GMRES projection relative residual %e in %d iterations.\n', relres, total_iter);
        fprintf('  GMRES cost %d sec\n\n', t_noEO);
    else
        fprintf('  GMRES failed to converge (flag = %d). Relative residual = %e.\n', flag, relres);
    end
    iters(k) = total_iter;
    nColors_noEO(k) = nColors;
    res_noEO{k} = resvec_noEO;
end
%% Multi-reordering with Even-Odd Reordering
fprintf('Check the EO and Coloring compatibility first...:\n')
evenCs_EO = zeros(1, k_total);
for k = 1:k_total
    [Colors, ncolor] = displacement_even_odd_coloring_nD_lattice(D, k, [0 0 0 0]);
    [isOK, evenCs, badCs, loc] = check_eo_compatibility_and_return_even(Colors, D, N_new);
    evenCs_EO(k) = length(unique(evenCs));
    fprintf('For k = %g, the number of colors = %g:\n', k, ncolor);
    if ~isOK
        fprintf('  Not compatible. Conflicting colors: '); fprintf('%d ', badCs); fprintf('\n');
        disp(loc{1});
        error('Aborted.')
    else
        fprintf('  OK: coloring is even/odd compatible.\n');
    end
end

% Then Solve A:
fprintf('Even-Odd reordering then multi-coloring:\n');
iters_eo = zeros(1, k_total);
res_eo = cell(1, k_total);
nColors_EO = zeros(1, k_total);
for k = 1:k_total
    % [Colors, nColors] = displacement_coloring_nD_lattice(D, k, p);
    [Colors, nColors] = displacement_even_odd_coloring_nD_lattice(D, k, p);
    Colors = kron(Colors, B_perm);
    [~, perm] = sort(Colors);
    A_perm = A(perm, perm);
    A_perm_gpu = gpuArray(A_perm);
    b_perm_gpu = b(perm);
    % colorView(A_perm, perm, colors, nColors, k);
    
    setup.type    = 'nofill';
    setup.droptol = 0;  
    [L, U] = ilu(A_perm, setup); % Matlab can't do ILU on GPU
    Lg = gpuArray(L); Ug = gpuArray(U);
    M_handle = @(x) Ug\(Lg\x);
    % x0 = b_perm;
    tic;
    [x_perm_gpu, flag, relres, iter, resvec_eo] = ...
        gmres(A_perm_gpu, b_perm_gpu, restart, tol, maxit, [], M_handle); % M1=Lg? M2=Ug?
    t_eo = toc;
    total_iter = (iter(1)-1)*restart + iter(2);
    relres_true_ = norm(b_perm_gpu - A_perm_gpu*x_perm_gpu)/norm(b_perm_gpu);
    
    if flag == 0
        fprintf('  When k = %d,', k);
        fprintf('  total colors = %d\n', nColors);
        fprintf('  the actual residual norm = %d\n', relres_true_);
        fprintf('  GMRES projection relative residual %e in %d iterations.\n', relres, total_iter);
        fprintf('  GMRES cost %d sec\n\n', t_eo);   
    else
        fprintf('  GMRES failed to converge (flag = %d). Relative residual = %e.\n', flag, relres);
    end
    iters_eo(k) = total_iter;
    res_eo{k} = resvec_eo;
    nColors_EO(k) = nColors;
end
% Undo permutation to original order
% invperm(perm) = 1:length(perm);
% x = x_perm(invperm);

%% Partial ILU(0)(A) with Schur Complement
fprintf('GMRES Schur Complement on GPU:\n')
evenCs_Schur = zeros(1, k_total);
for k = 1:k_total
    [Colors, ncolor] = displacement_even_odd_coloring_nD_lattice(D, k, [0 0 0 0]);
    [isOK, evenCs, badCs, loc] = check_eo_compatibility_and_return_even(Colors, D, N_new);
    evenCs_Schur(k) = length(unique(evenCs));
    fprintf('For k = %g, the number of colors = %g:\n', k, ncolor);
    if ~isOK
        fprintf('  Not compatible. Conflicting colors: '); fprintf('%d ', badCs); fprintf('\n');
        disp(loc{1});
        error('Aborted.')
    else
        fprintf('  OK: coloring is even/odd compatible.\n');
    end
end

iters_sch = zeros(1, k_total);
n = N_new;

resvec_even = cell(1, k_total);
nColors_Schur = zeros(1, k_total);
for k = 1:k_total 
    [colors, nColors] = displacement_even_odd_coloring_nD_lattice(D, k, p);
    colors = kron(colors, B_perm);
    [~, perm] = sort(colors);
    A_perm = A(perm, perm);
    b_perm_gpu = bg(perm);
    % colorView(A_perm, perm, colors, nColors, k);

    ap_ee = A_perm(1:n/2,1:n/2);          ap_ee = gpuArray(ap_ee);
    ap_eo = A_perm(1:n/2, n/2+1:end);     ap_eo = gpuArray(ap_eo);
    ap_oe = A_perm(n/2+1:end, 1:n/2);     ap_oe = gpuArray(ap_oe);

    ap_oo = A_perm(n/2+1:end, n/2+1:end); ap_oo = gpuArray(ap_oo);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % % Build 64x64xnb on CPU
    % nb=size(ap_oo,1)/bs;
    % AooPages = zeros(bs,bs,nb,'double');
    % for b = 1:nb 
    %     idx = (b-1)*bs+1 : b*bs;                           % On CPU
    %     AooPages(:,:,b) = full(A_perm(n/2+idx, n/2+idx));  % dense small blocks
    % end
    % AooPages_g = gpuArray(AooPages);                       % cp2gpu
    % 
    % % solve_oo_gpu(y) = inv(A_oo) y
    % solve_oo_gpu = @(yg) ...
    %     reshape(pagefun(@mldivide, AooPages_g, reshape(yg, bs,1,nb)), ...
    %     [], 1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    solve_oo_gpu = @(xg) ap_oo\xg; % normal
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Sparse MLDIVIDE only supports sparse square matrices      %
    %  divided by full column vectors.                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % === Schur Complement === %
    % s_ee_gpu = @(xg)... 
    %   ap_ee*xg - ap_eo * gpuArray( solve_ap_oo(gather(ap_oe*xg)) ); % IMPLICITLY, do the inv ap_oo in advance, inv(ap_oo) should be also diag not dense.

    s_ee_gpu = @(xg)...
       ap_ee*xg - ap_eo * solve_oo_gpu(ap_oe*xg);

    % Eliminate odd -> GMRES solve for even
    bp_e_gpu = b_perm_gpu(1:n/2);
    bp_o_gpu = b_perm_gpu(n/2+1 : end);
    % rhs_o = bp_o - ap_oe * (ap_ee \ bp_e);
    rhs_e_gpu = bp_e_gpu - ap_eo * solve_oo_gpu(bp_o_gpu);

    % === Preconditioned Handle (Left Preconditioning) === %
    setup.type    = 'nofill';
    setup.droptol = 0;  
    [L, U] = ilu(A_perm, setup);      % on CPU
    M = L*U; 
    M_ee_old = M(1:n/2, 1:n/2);       % "Cut" on CPU
    q_M = colamd(M_ee_old);
    M_ee = M_ee_old(:, q_M);

    % M_ee_g = full(gpuArray(M_ee));  % NOOOO, memory boom!
    
    % === AK(K^{-1}x)=y (Right Precondtioning) -> AKt=y -> x=Kt ===
    % M_ee, M_oo are pure block diagonal matrices, no zero pivot
    [LM_ee, UM_ee] = lu(M_ee);  % intro fill-ins but acc GMRES
    M_handle = @(xg) gpuArray(UM_ee\(LM_ee\gather(xg))); % acc 100% in sec  

    M_handle_old = @(xg) gpuArray(M_ee_old)\xg;
    %%%%%%%%%%%%%%%%%%%%%%%%% Timers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % xg = randn(n/2,1,'gpuArray');
    % t1 = gputimeit(@() ap_ee*xg);
    % t2 = gputimeit(@() ap_oe*xg);
    % t3 = timeit(@() solve_ap_oo(randn(n/2,1)));          
    % t4 = gputimeit(@() s_ee_gpu(xg));
    % t5 = gputimeit(@() M_handle(xg));
    % 
    % fprintf(['  SpMV ee: %.3f ms | SpMV oe: %.3f ms | CPU solve_oo: %.3f ms |' ...
    %     ' S*x: %.3f ms | M*x: %.3f ms\n\n'],...
    %         t1*1e3, t2*1e3, t3*1e3, t4*1e3, t5*1e3);
    %%%%%%%%%%%%%%%%%%%%%%%%%% Timers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fprintf('  GMRES Start...\n');
    tic;
    [x_even_gpu, flag, relres_even, iter_even, resvec_even{k}] = ...
        gmres(s_ee_gpu, rhs_e_gpu, restart, tol, maxit, [], M_handle); 
    t_imp = toc;
    
    total_iter = (iter_even(1)-1)*restart + iter_even(2);
    relres_even_true = norm(rhs_e_gpu - s_ee_gpu(x_even_gpu))/norm(rhs_e_gpu);
    x_odd = ap_oo \ (bp_o_gpu - ap_oe * x_even_gpu);

    iters_sch(k) = total_iter;
    nColors_Schur(k) = nColors;
    fprintf('  When k = %d,', k)
    fprintf('  Total colors = %d.\n', nColors);
    fprintf('  GMRES projection relative residual %e in %d iterations.\n', relres_even_true, total_iter);
    fprintf('  GMRES w/ imp Schur cost %d sec\n\n', t_imp);
     
    % === Combine x_even and x_odd ===
end
%% GMRES Schur Complement w/o preconditioner
fprintf('GMRES Schur Complement no precondtioner:\n')
iters_sch_no_prec = zeros(1, k_total);
n = N_new;

resvec_even_no_prec = cell(1, k_total);
nColors_Schur = zeros(1, k_total);
for k = 1:k_total 
    [colors, nColors] = displacement_even_odd_coloring_nD_lattice(D, k, p);
    colors = kron(colors, B_perm);
    [~, perm] = sort(colors);
    A_perm = A(perm, perm);
    b_perm_gpu = bg(perm);

    ap_ee = A_perm(1:n/2,1:n/2);          ap_ee = gpuArray(ap_ee);
    ap_eo = A_perm(1:n/2, n/2+1:end);     ap_eo = gpuArray(ap_eo);
    ap_oe = A_perm(n/2+1:end, 1:n/2);     ap_oe = gpuArray(ap_oe);

    ap_oo = A_perm(n/2+1:end, n/2+1:end); ap_oo = gpuArray(ap_oo);
 
    solve_oo_gpu = @(xg) ap_oo\xg; % normal

    s_ee_gpu = @(xg)...
       ap_ee*xg - ap_eo * solve_oo_gpu(ap_oe*xg);

    % Eliminate odd -> GMRES solve for even
    bp_e_gpu = b_perm_gpu(1:n/2);
    bp_o_gpu = b_perm_gpu(n/2+1 : end);
    % rhs_o = bp_o - ap_oe * (ap_ee \ bp_e);
    rhs_e_gpu = bp_e_gpu - ap_eo * solve_oo_gpu(bp_o_gpu);

    fprintf('  GMRES Start...\n');
    tic;
    [x_even_gpu, flag, relres_even_no, iter_even, resvec_even_no_prec{k}] = ...
        gmres(s_ee_gpu, rhs_e_gpu, restart, tol, maxit, [], []); 
    t_imp = toc;

    total_iter = (iter_even(1)-1)*restart + iter_even(2);
    x_odd = ap_oo \ (bp_o_gpu - ap_oe * x_even_gpu);
    relres_even_no_true = norm(rhs_e_gpu - s_ee_gpu(x_even_gpu))/norm(rhs_e_gpu);
    iters_sch_no_prec(k) = total_iter;
    nColors_Schur(k) = nColors;

    fprintf('  When k = %d,', k)
    fprintf('  Total colors = %d.\n', nColors);
    fprintf('  GMRES projection relative residual %e in %d iterations.\n', relres_even_no_true, total_iter);
    fprintf('  GMRES w/ imp Schur cost %d sec\n\n', t_imp);
     
    % Combine x_even and x_odd
end

%% Plot all Convergence
figure; 
for k = 1:k_total

    semilogy(res_noEO{k},'--' ,'LineWidth', 1.2, 'DisplayName', sprintf('w/o EO, k = %d', k));
    semilogy(res_eo{k},'-', 'LineWidth', 1.2, 'DisplayName', sprintf('w/ EO, k = %d', k));
    
    semilogy(resvec_even_no_prec{k}, 'LineWidth', 2, 'DisplayName', sprintf('w/ EO, no ilu(0), k = %d', k));
    semilogy(resvec_even{k},'-', 'LineWidth', 1.2, 'DisplayName', sprintf('w/ EO, k = %d', k));
    hold on;
end
semilogy(resvec_GMRES,'-', 'LineWidth', 1.2, 'DisplayName', sprintf('GMRES'));
semilogy(resvec_ilu0,'-', 'LineWidth', 1.2, 'DisplayName', sprintf('ILU(0) Natural'));

xlabel('Iteration');
ylabel('Residual Norm');
yline(tol,'r--','DisplayName', sprintf('Tol'));
title('Schur Comp. GMRES Residual Convergence with EO');
legend show;
grid on;
%%
figure; subplot(1,2,1);
X = 1:k_total;
Y = [iters(:), iters_eo(:), iters_sch(:), iters_sch_no_prec(:)];
total_colors = [nColors_noEO(:), nColors_EO(:), nColors_Schur(:), nColors_Schur(:)];
even_colors = [evenCs_noEO(:), evenCs_EO(:), evenCs_Schur(:), evenCs_Schur(:)];
bar(X, Y, 'grouped'); hold on;

yline(total_iter_pure, '--r', 'Pure GMRES', 'LineWidth', 2);
yline(total_iter_ilu0, '--b', 'ILU(0) Natural', 'LineWidth', 2);

xlabel('k');
ylabel('GMRES Iterations');
legend('Without EO','With EO','Schur Comp.', 'Schur Comp. w/o ILU(0)','Pure GMRES','ILU(0) Natural','Location','northwest');
title('GMRES Iterations vs. k');
grid on;

subplot(1,2,2);
% bar(X, total_colors, 'grouped');
bar(X, even_colors, 'grouped');
xlabel('k');
ylabel('Number of Even Colors');
legend('Without EO', 'With EO', 'Schur Comp.', 'Schur Comp. w/o ILU(0)', 'Location', 'northwest');

p_str = strtrim(sprintf('%g ', p));
title(sprintf('Number of Even Colors vs. k,  Disp = [%s]', p_str));
grid on;

%%
function colorView(A, order, colors, nColors, k)
figure;
N = size(A, 1);
% Reording by Colors
[~, order] = sort(colors);     
newPos = zeros(N,1);
newPos(order) = 1:N;          
[rowIdx, colIdx] = find(spones(A));
rowIdxNew = newPos(rowIdx);
colIdxNew = newPos(colIdx);


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
grid on;
end