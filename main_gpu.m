% 08/01/2025
% Problems:
%  Preconditioners of GPU_GMRES in the Schur Complement section
%  
%% Testing the followings:
%
%   Permute matrix with multicoloring, even-odd reordering, 
%   apply ILU(0) preconditioner, 
%   the last part for implementing the Schur complement for solving Ax = b.
%
%  Main args:
%    D       : length N vector for generating Nd matrix
%    p       : length N vector for displacement at each d
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
% figure; spy(A); title("Original A");
N = size(A,1);
% A = A + speye(N)*1e-7; % non-singular
%A = A.*(1+rand(N,N)/10) + speye(N)*1e-7;
rng(1);
A = kron(A, (rand(1)));   % For denser "boxed" diag
B_perm = ones(1,1);

N_new = size(A, 1);
A = A + speye(N_new)*1e-7; % non-singular

Ag = gpuArray(A); % GPU
% figure; spy(Ag); title("After Kron(A, rand(48))")

b = ones(N_new, 1);  % RHS: all-ones vector
bg = randn(N_new, 1);
% bg = randn(N_new, 1, "gpuArray");
% %
maxit = 1000;
restart = 100;
% x0 = b;
%% "Pure Iterations"
[xg, flag, relres, iter, resvec] = ... % 
    gmres(Ag, bg, restart, tol, maxit, [], []);
relres_true = norm(bg - Ag*xg)/norm(bg);
fprintf('Pure iterative results w/o preconditioner:\n');
fprintf('  The actual residual norm = %d\n', relres_true);
fprintf('  GMRES projection relative residual %e in %d iterations.\n\n', relres, sum(iter));

%% Use ILU(0) Preconditioner Only
fprintf('Use ilu(0) but no multi-reordering:\n');
setup.type    = 'nofill';
setup.droptol = 0;  
[L, U] = ilu(A, setup); % Matlab can't do ILU on GPU
Lg = gpuArray(L); Ug = gpuArray(U);
% M = Lg*Ug;
M_handle = @(x) Ug\(Lg\x);

% x0 = bg;
[x_perm_gpu, flag, relres, iter, resvec] = ...
    gmres(Ag, bg, restart, tol, maxit, [], M_handle); % Handle
relres_true = norm(bg - Ag*x_perm_gpu)/norm(bg);

fprintf('  The actual residual norm = %d\n', relres_true);
fprintf('  GMRES projection relative residual %e in %d iterations.\n\n', relres, sum(iter));
%% Use muticoloring only without Even-Odd ordering
fprintf('Only multi-coloring w/o Even-Odd:\n');
iters = zeros(1, k_total);
relres_true = cell(1, k_total);

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
    
    [x_perm_gpu, flag, relres, iter, resvec] = ...
        gmres(A_perm_gpu, b_perm_gpu, restart, tol, maxit, [], M_handle);
    
    relres_true_ = norm(b_perm_gpu - A_perm_gpu*x_perm_gpu)/norm(b_perm_gpu);
    
    if flag == 0
        fprintf('  When k = %d,', k);
        fprintf('  total colors = %d\n', nColors);
        fprintf('  the actual residual norm = %d\n', relres_true_);
        fprintf('  GMRES projection relative residual %e in %d iterations.\n\n', relres, sum(iter));
        
    else
        fprintf('  GMRES failed to converge (flag = %d). Relative residual = %e.\n', flag, relres);
    end
    iters(k) = sum(iter);
    relres_true{k} = resvec/norm( U\b_perm_gpu );
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
    A_perm_gpu = gpuArray(A_perm);  % Copy to GPU
    b_perm_gpu = b(perm);
    % colorView(A_perm, perm, colors, nColors, k);
    
    setup.type    = 'nofill';
    setup.droptol = 0;  
    [L, U] = ilu(A_perm, setup); % Matlab can't do ILU on GPU
    Lg = gpuArray(L); Ug = gpuArray(U);
    
    % x0 = b_perm;
    
    [x_perm_gpu, flag, relres, iter, resvec] = ...
        gmres(A_perm_gpu, b_perm_gpu, restart, tol, maxit, Lg, Ug); % M1=Lg? M2=Ug?
    
    relres_true_ = norm(b_perm_gpu - A_perm_gpu*x_perm_gpu)/norm(b_perm_gpu);
    
    if flag == 0
        fprintf('  When k = %d,', k);
        fprintf('  total colors = %d\n', nColors);
        fprintf('  the actual residual norm = %d\n', relres_true_);
        fprintf('  GMRES projection relative residual %e in %d iterations.\n\n', relres, sum(iter));
    
    else
        fprintf('  GMRES failed to converge (flag = %d). Relative residual = %e.\n', flag, relres);
    end
    iters_eo(k) = sum(iter);
    relres_true_eo{k} = resvec/norm( U\b_perm_gpu );
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

figure; clf;
for k = 1:k_total
    semilogy(relres_true{k},'--' ,'LineWidth', 1.2, 'DisplayName', sprintf('w/o EO, k = %d', k));
    hold on;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xlabel('Iteration');
ylabel('Residual Norm');
yline(tol,'r--');
title('GMRES Residual w/o EO');
legend show;
grid on;

figure;
clf;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:k_total
    semilogy(relres_true_eo{k},'-', 'LineWidth', 1.2, 'DisplayName', sprintf('w/ EO, k = %d', k));
    hold on;
end
xlabel('Iteration');
ylabel('Residual Norm');
yline(tol,'r--','DisplayName', sprintf('Tol'));
title('GMRES Residual Convergence (With vs. Without EO)');
legend show;
grid on;

%% Partial ILU(0)(A) with Schur Complement
fprintf('Schur Complement GPU ver:\n')
iters_sch = zeros(1, k_total);
n = N_new;

% M_ee=M(1:n/2,1:n/2); %
resvec_even = cell(1, k_total);
for k = 1:k_total 
    [colors, nColors] = displacement_even_odd_coloring_nD_lattice(D, k, p);
    colors = kron(colors, B_perm);
    [~, perm] = sort(colors);
    A_perm = A(perm, perm);
    b_perm = b(perm);
    b_perm_gpu = bg(perm);
    % colorView(A_perm, perm, colors, nColors, k);

    ap_ee = A_perm(1:n/2,1:n/2);         ap_ee = gpuArray(ap_ee);
    ap_eo = A_perm(1:n/2, n/2+1:end);    ap_eo = gpuArray(ap_eo);
    ap_oe = A_perm(n/2+1:end, 1:n/2);    ap_oe = gpuArray(ap_oe);
    
    ap_oo = A_perm(n/2+1:end, n/2+1:end);
    [l_cpu, u_cpu, p_oo, q_oo] = lu(ap_oo);
    solve_ap_oo = @(y) q_oo*(u_cpu \ (l_cpu \ (p_oo*y))); % on cpu
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Sparse MLDIVIDE only supports sparse square matrices      %
    %  divided by full column vectors.                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % === Schur Complement === %
    s_ee_gpu = @(xg)... 
       ap_ee*xg - ap_eo * gpuArray( solve_ap_oo(gather(ap_oe*xg)) ); % IMPLICITLY, do the inv ap_oo in advance, inv(ap_oo) should be also diag not dense.
    
    % Eliminate odd -> GMRES solve for even
    bp_e_gpu = b_perm_gpu(1:n/2);
    bp_e = b_perm(1:n/2);
    bp_o_gpu = b_perm_gpu(n/2+1 : end);
    bp_o = b_perm(n/2+1 : end);
    rhs_e_gpu = bp_e_gpu - ap_eo * gpuArray(solve_ap_oo(bp_o));
    % rhs_o = bp_o - ap_oe * (ap_ee \ bp_e);
    
    % === Preconditioned Handle (Left Preconditioning) === %
    setup.type    = 'nofill';
    setup.droptol = 0;  
    [L, U] = ilu(A_perm, setup);       % on CPU
    M = L*U; M_ee = M(1:n/2, 1:n/2);   % "Cut" on CPU
    M_ee_g = full(gpuArray(M_ee));
    [LM_ee_g, UM_ee_g] = lu(M_ee_g); % on GPU, no zero pivot I guess?
    % AK(K^{-1}x)=y (Right Precondtioning) -> AKt=y -> x=Kt
    M_handle = @(xg) UM_ee_g\(LM_ee_g\xg);
    
    tic;
    M_handle_old = @(xg) M_ee_g\xg;
    % s_ee_mul = @(x)ap_ee*x - ap_eo * (ap_oo_inv*(ap_oe*x)); 
    % x0 = bp_e_t;
    [x_even_gpu, flag, relres_even, iter_even, resvec_even{k}] = ...
        gmres(s_ee_gpu, rhs_e_gpu, restart, tol, maxit, [], M_handle); 
    t_imp = toc;

    x_odd = ap_oo \ (bp_o_gpu - ap_oe * x_even_gpu);
    iters_sch(k) = sum(iter_even);
    
    fprintf('  When k = %d,', k)
    fprintf('  Total colors = %d.\n', nColors);
    fprintf('  GMRES projection relative residual %e in %d iterations.\n', relres_even, sum(iter_even));
    fprintf('  GMRES w/ imp Schur used %d sec\n', t_imp);
     
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

figure;
plot(1:k_total, iters, '-o', 'LineWidth', 2); hold on;
plot(1:k_total, iters_eo, '-s', 'LineWidth', 2);hold on;
plot(1:k_total, iters_sch, '-s', 'LineWidth', 2);
xlabel('k'); ylabel('GMRES Iterations');
legend('Without EO', 'With EO','Schur Comp.','Location', 'northwest');
title('GMRES Iterations vs. k');
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
