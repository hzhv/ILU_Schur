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
figure; spy(A); title("Original A");
N = size(A,1);
% A = A + speye(N)*1e-7; % non-singular
%A = A.*(1+rand(N,N)/10) + speye(N)*1e-7;
rng(1);
A = kron(A, (rand(48)));   % For denser "boxed" diag
B_perm = ones(48,1);

N_new = size(A, 1);
A = A + speye(N_new)*1e-7; % non-singular

Ag = gpuArray(A); % GPU

% figure; spy(Ag); title("After Kron(A, rand(48))")
%%
% b = ones(N_new, 1);  % RHS: all-ones vector
% bg = randn(N_new, 1);
bg = randn(N_new, 1, "gpuArray");

maxit = 300;
% restart = size(Ag, 1);
restart = 200;
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

maxit = size(Ag,1);
restart = size(Ag, 1);
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

    maxit = size(A_perm_gpu, 1);
    restart = size(A_perm_gpu, 1);
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
    b_perm_gpu = bg(perm);
    % colorView(A_perm, perm, colors, nColors, k);
    
    setup.type    = 'nofill';
    setup.droptol = 0;  
    [L, U] = ilu(A_perm, setup); % Matlab can't do ILU on GPU
    Lg = gpuArray(L); Ug = gpuArray(U);
    
    maxit = size(A_perm,1);
    restart = size(A_perm, 1);
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
fprintf('Schur Complement:\n')
iters_sch = zeros(1, k_total);
n = N;
% M = l*u; 
% M_ee=M(1:n/2,1:n/2); %
resvec_even = cell(1, k_total);
for k = 1:k_total 
    [colors, nColors] = displacement_even_odd_coloring_nD_lattice(D, k, p);
    colors = kron(colors, B_perm);
    [~, perm] = sort(colors);
    A_perm = A(perm, perm);
    A_perm_gpu = gpuArray(A_perm);  % Copy to GPU
    b_perm_gpu = bg(perm);
    % colorView(A_perm, perm, colors, nColors, k);
    
    setup.type    = 'nofill';
    setup.droptol = 0;  
    [L, U] = ilu(A_perm, setup);
    M = L*U; 
    M = M(1:n/2, 1:n/2); % "Cut" on CPU
    Mg = gpuArray(M);    % then copy to GPU
    % AK(K^{-1}x)=y (Right Precondtioning) -> AKt=y -> x=Kt
    M_handle = @(x) Mg\x;

    % All "Cut" on CPU
    ap_ee = A_perm(1:n/2,1:n/2);         ap_ee = gpuArray(ap_ee);
    ap_oo = A_perm(n/2+1:end,n/2+1:end); ap_oo = gpuArray(ap_oo);
    ap_eo = A_perm(1:n/2, n/2+1:end);    ap_eo = gpuArray(ap_eo);
    ap_oe = A_perm(n/2+1:end, 1:n/2);    ap_oe = gpuArray(ap_oe);
    
    bp_e = b_perm_gpu(1:n/2);
    bp_o = b_perm_gpu(n/2+1 : end);
    
    % rhs_o = bp_o - ap_oe * (ap_ee \ bp_e);
    rhs_e = bp_e - ap_eo * (ap_oo \ bp_o);

    % Eliminate odd -> GMRES solve for even
    maxit = size(ap_ee,1);
    restart = size(ap_ee, 1);
    
    % % EXPLICITLY
    % tic;
    % s_ee_mul = ap_ee - ap_eo * (ap_oo \ ap_oe);       
    % [x_even_gpu, flag, relres_even, iter_even, resvec_even{k}] = ...
    %     gmres(s_ee_mul, rhs_e, restart, tol, maxit, [], M);
    % t_exp = toc;
    % fprintf('GMRES w/ exp Schur:%d', t_exp);
    % % figure; clf; spy(s); title(sprintf('S_{even} for k=%d', k)); grid on; 
    
    % Preconditioned Handle (Left Preconditioning)
    
    s_ee_mul = @(x)...
        ap_ee*x - ap_eo * (ap_oo \ (ap_oe*x)); % IMPLICITLY, do the inv ap_oo in advance, inv(ap_oo) should be also diag not dense.
    tic;
    
    % s_ee_mul = @(x)ap_ee*x - ap_eo * (ap_oo_inv*(ap_oe*x)); 
    % x0 = bp_e_t;
    [x_even_gpu, flag, relres_even, iter_even, resvec_even{k}] = ...
        gmres(s_ee_mul, rhs_e, restart, tol, maxit, [], M_handle); 
    t_imp = toc;
    
    fprintf('  When k = %d,', k)
    fprintf('  Total colors = %d.\n', nColors);
    fprintf('  GMRES projection relative residual %e in %d iterations.\n', relres_even, sum(iter_even));
    fprintf('  GMRES w/ imp Schur used %d sec\n', t_imp);
    x_odd = ap_oo \ (bp_o - ap_oe * x_even_gpu);
    iters_sch(k) = sum(iter_even);
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

%% LOG
% NO Kron, NO GPU ACC:  D = [8 8 8 16]; p = [0 0 0 0]; Tol = 1e-6
%   When k = 1,  Total colors = 2
%   GMRES projection relative residual 7.138174e-07 in 32 iterations.
%   GMRES w/ exp Schur used 1.524670e-02 sec
%   GMRES w/ imp Schur used 1.380720e-02 sec
% 
%   When k = 2,  Total colors = 21
%   GMRES projection relative residual 4.437670e-07 in 36 iterations.
%   GMRES w/ exp Schur used 2.028571e+00 sec
%   GMRES w/ imp Schur used 1.063467e+00 sec
% 
%   When k = 3,  Total colors = 16
%   GMRES projection relative residual 8.489670e-07 in 32 iterations.
%   GMRES w/ exp Schur used 3.434697e-01 sec
%   GMRES w/ imp Schur used 2.094873e-01 sec
% 
% GPU ACC no Preconditioners:  
%   When k = 1,  Total colors = 2.
%   GMRES projection relative residual 5.859539e-07 in 32 iterations.
%   GMRES w/ imp Schur used 1.819638e-01 sec
% 
%   When k = 2,  Total colors = 16.
%   GMRES projection relative residual 5.859539e-07 in 32 iterations.
%   GMRES w/ imp Schur used 2.089628e-01 sec
% 
%   When k = 3,  Total colors = 16.
%   GMRES projection relative residual 5.859539e-07 in 32 iterations.
%   GMRES w/ imp Schur used 1.408641e-01 sec

% GPU ACC w/ Preconditioned S:
%   When k = 1,  Total colors = 2.
%   GMRES projection relative residual 7.412678e-07 in 32 iterations.
%   GMRES w/ imp Schur used 1.618779e-01 sec
%   When k = 2,  Total colors = 16.
%   GMRES projection relative residual 7.412678e-07 in 32 iterations.
%   GMRES w/ imp Schur used 1.556399e-01 sec
%   When k = 3,  Total colors = 16.
%   GMRES projection relative residual 7.412678e-07 in 32 iterations.
%   GMRES w/ imp Schur used 1.555296e-01 sec
%% 
% No Kron
% Pure iterative results w/o preconditioner:
%   The actual residual norm = 7.957382e-10
%   GMRES projection relative residual 4.174901e-11 in 24 iterations.
% 
% Only multi-coloring w/o Even-Odd:
%   When k = 1,  total colors = 2.
%   the actual residual norm = 4.899220e-10
%   GMRES projection relative residual 4.386977e-13 in 14 iterations.
% 
%   When k = 2,  total colors = 16.
%   the actual residual norm = 1.840195e-07
%   GMRES projection relative residual 1.796352e-07 in 22 iterations.
% 
%   When k = 3,  total colors = 16.
%   the actual residual norm = 5.990281e-07
%   GMRES projection relative residual 7.445310e-07 in 20 iterations.
% 
% Even-Odd reordering then multi-coloring:
%   When k = 1,  total colors = 2.
%   the actual residual norm = 4.899220e-10
%   GMRES projection relative residual 4.386977e-13 in 14 iterations.
% 
%   When k = 2,  total colors = 16.
%   the actual residual norm = 3.883715e-10
%   GMRES projection relative residual 2.507538e-12 in 14 iterations.
% 
%   When k = 3,  total colors = 16.
%   the actual residual norm = 3.883715e-10
%   GMRES projection relative residual 2.507538e-12 in 14 iterations.
% 
%   
%% 08/12/2025 Kron(rand(48))
% 
% MATLAB crash file:/home/hli31/matlab_crash_dump.3653661-1:
% 
% 
% --------------------------------------------------------------------------------
%        Segmentation violation detected at Tue Aug 12 17:07:20 2025 -0400
% --------------------------------------------------------------------------------
% 
% Configuration:
%   Crash Decoding           : Disabled - No sandbox or build area path
%   Crash Mode               : continue (default)
%   Default Encoding         : UTF-8
%   Deployed                 : false
%   GNU C Library            : 2.35 stable
%   Graphics Driver          : Unknown software 
%   Graphics card 1          : 0x1a03 ( 0x1a03 ) 0x2000 Version 0.0.0.0 (0-0-0)
%   Graphics card 2          : 0x10de ( 0x10de ) 0x20f1 Version 570.133.7.0 (0-0-0)
%   Java Version             : Java 1.8.0_181-b13 with Oracle Corporation Java HotSpot(TM) 64-Bit Server VM mixed mode
%   MATLAB Architecture      : glnxa64
%   MATLAB Entitlement ID    : 810081
%   MATLAB Root              : /opt/MATLAB/R2019a
%   MATLAB Version           : 9.6.0.1072779 (R2019a)
%   OpenGL                   : software
%   Operating System         : Linux 5.15.0-142-generic #152-Ubuntu SMP Mon May 19 10:54:31 UTC 2025 x86_64
%   Process ID               : 3653661
%   Processor ID             : x86 Family 143 Model 49 Stepping 0, AuthenticAMD
%   Session Key              : 15f40013-1a32-4662-a4a7-478a7e59f059
%   Static TLS mitigation    : Enabled: Full
%   Window System            : Moba/X (12101015), display localhost:11.0
% 
% Fault Count: 1
% 
% 
% Abnormal termination:
% Segmentation violation
% 
% Register State (from fault):
%   RAX = 0000000000000000  RBX = ffffffff80003946
%   RCX = 00007ef60c9fa014  RDX = 00000000000000c0
%   RSP = 00007f003dff9d00  RBP = 00007ef60cb7a010
%   RSI = 0000000000000003  RDI = 0000000000000001
% 
%    R8 = 00007edc4147f528   R9 = 00007ef688cb8dd0
%   R10 = 00007ef68a5eef40  R11 = 0000000000013980
%   R12 = 0000000000006004  R13 = 0000000000000000
%   R14 = 00007ef689e6ef00  R15 = 00007ef60c9fa010
% 
%   RIP = 00007efa2526d24f  EFL = 0000000000010293
% 
%    CS = 0033   FS = 0000   GS = 0000
% 
% Stack Trace (from fault):
% [  0] 0x00007efa2526d24f /opt/MATLAB/R2019a/bin/glnxa64/libcusolver.so.10.0+00447055
% [  1] 0x00007efa2526ab25 /opt/MATLAB/R2019a/bin/glnxa64/libcusolver.so.10.0+00437029 cusolverSpXcsrqrAnalysisHost+00006821
% [  2] 0x00007efa252f59c6 /opt/MATLAB/R2019a/bin/glnxa64/libcusolver.so.10.0+01006022 cusolverSpXcsrqrAnalysis+00001222
% [  3] 0x00007efa2531df94 /opt/MATLAB/R2019a/bin/glnxa64/libcusolver.so.10.0+01171348
% [  4] 0x00007efca63d4221   /opt/MATLAB/R2019a/bin/glnxa64/libmwgpusparse.so+00033313 _ZN9gpusparse6splsmvIdEEvP11CUstream_stPT_RKNS_8CsrArrayIKS3_EEPS6_diPi+00000241
% [  5] 0x00007efa435f61aa         /opt/MATLAB/R2019a/bin/glnxa64/libmwgpu.so+12370346
% [  6] 0x00007efa435f74e4         /opt/MATLAB/R2019a/bin/glnxa64/libmwgpu.so+12375268
% [  7] 0x00007efa435f8480         /opt/MATLAB/R2019a/bin/glnxa64/libmwgpu.so+12379264
% [  8] 0x00007efa42de840c         /opt/MATLAB/R2019a/bin/glnxa64/libmwgpu.so+03925004
% [  9] 0x00007efa42e43070         /opt/MATLAB/R2019a/bin/glnxa64/libmwgpu.so+04296816
% [ 10] 0x00007efa42d3795d         /opt/MATLAB/R2019a/bin/glnxa64/libmwgpu.so+03201373
% [ 11] 0x00007f003d50b818   /opt/MATLAB/R2019a/bin/glnxa64/libmwmcos_impl.so+03954712
% [ 12] 0x00007f003d50e828   /opt/MATLAB/R2019a/bin/glnxa64/libmwmcos_impl.so+03967016
% [ 13] 0x00007f003d5159c2   /opt/MATLAB/R2019a/bin/glnxa64/libmwmcos_impl.so+03996098
% [ 14] 0x00007f003d50a741   /opt/MATLAB/R2019a/bin/glnxa64/libmwmcos_impl.so+03950401
% [ 15] 0x00007f003d685f4f   /opt/MATLAB/R2019a/bin/glnxa64/libmwmcos_impl.so+05504847
% [ 16] 0x00007f003d67f3a7   /opt/MATLAB/R2019a/bin/glnxa64/libmwmcos_impl.so+05477287
% [ 17] 0x00007f004bf2ce16 /opt/MATLAB/R2019a/bin/glnxa64/libmwm_dispatcher.so+00564758
% [ 18] 0x00007f004bf2d161 /opt/MATLAB/R2019a/bin/glnxa64/libmwm_dispatcher.so+00565601 _ZN18Mfh_MATLAB_fn_impl8dispatchEiPSt10unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEiPPS1_+00000033
% [ 19] 0x00007f004512627a       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+13877882
% [ 20] 0x00007f004512bc0f       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+13900815
% [ 21] 0x00007f0045223f91       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+14917521
% [ 22] 0x00007f0045224048       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+14917704
% [ 23] 0x00007f0045190664       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+14313060
% [ 24] 0x00007f00451b7d7d       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+14474621
% [ 25] 0x00007f004494868b       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+05629579
% [ 26] 0x00007f004494a724       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+05637924
% [ 27] 0x00007f004494766d       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+05625453
% [ 28] 0x00007f004493c211       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+05579281
% [ 29] 0x00007f004493c449       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+05579849
% [ 30] 0x00007f0044946e76       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+05623414
% [ 31] 0x00007f0044946f76       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+05623670
% [ 32] 0x00007f0044a7ecd9       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+06900953
% [ 33] 0x00007f0044a82413       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+06915091
% [ 34] 0x00007f0044fe5d61       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+12565857
% [ 35] 0x00007f0045114b01       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+13806337
% [ 36] 0x00007f0045114c7d       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+13806717
% [ 37] 0x00007f004bfab57f /opt/MATLAB/R2019a/bin/glnxa64/libmwm_dispatcher.so+01082751 _ZN8Mfh_file20dispatch_file_commonEMS_FviPP11mxArray_tagiS2_EiS2_iS2_+00000207
% [ 38] 0x00007f004bfad07e /opt/MATLAB/R2019a/bin/glnxa64/libmwm_dispatcher.so+01089662
% [ 39] 0x00007f004bfad5c1 /opt/MATLAB/R2019a/bin/glnxa64/libmwm_dispatcher.so+01091009 _ZN8Mfh_file8dispatchEiPSt10unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEiPPS1_+00000033
% [ 40] 0x00007f004512627a       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+13877882
% [ 41] 0x00007f004512bc0f       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+13900815
% [ 42] 0x00007f00452240ec       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+14917868
% [ 43] 0x00007f0045190664       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+14313060
% [ 44] 0x00007f00451b708d       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+14471309
% [ 45] 0x00007f004494868b       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+05629579
% [ 46] 0x00007f004494a724       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+05637924
% [ 47] 0x00007f004494766d       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+05625453
% [ 48] 0x00007f004493c211       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+05579281
% [ 49] 0x00007f004493c449       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+05579849
% [ 50] 0x00007f0044946e76       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+05623414
% [ 51] 0x00007f0044946f76       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+05623670
% [ 52] 0x00007f0044a7ecd9       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+06900953
% [ 53] 0x00007f0044a82413       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+06915091
% [ 54] 0x00007f0044fe5d61       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+12565857
% [ 55] 0x00007f0044f8f67c       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+12211836
% [ 56] 0x00007f0044f93baf       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+12229551
% [ 57] 0x00007f0044f96eb2       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+12242610
% [ 58] 0x00007f004503470f       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+12887823
% [ 59] 0x00007f00450349fa       /opt/MATLAB/R2019a/bin/glnxa64/libmwm_lxe.so+12888570
% [ 60] 0x00007f004ce5337e      /opt/MATLAB/R2019a/bin/glnxa64/libmwbridge.so+00283518 _Z8mnParserv+00000590
% [ 61] 0x00007f004c3674f2         /opt/MATLAB/R2019a/bin/glnxa64/libmwmcr.so+01012978
% [ 62] 0x00007f0054bdf96b         /opt/MATLAB/R2019a/bin/glnxa64/libmwmvm.so+03672427 _ZN14cmddistributor15PackagedTaskIIP10invokeFuncIN7mwboost8functionIFvvEEEEENS2_10shared_ptrINS2_13unique_futureIDTclfp_EEEEEERKT_+00000059
% [ 63] 0x00007f0054bdfa58         /opt/MATLAB/R2019a/bin/glnxa64/libmwmvm.so+03672664 _ZNSt17_Function_handlerIFN7mwboost3anyEvEZN14cmddistributor15PackagedTaskIIP10createFuncINS0_8functionIFvvEEEEESt8functionIS2_ET_EUlvE_E9_M_invokeERKSt9_Any_data+00000024
% [ 64] 0x00007f004c795dcc         /opt/MATLAB/R2019a/bin/glnxa64/libmwiqm.so+00769484 _ZN7mwboost6detail8function21function_obj_invoker0ISt8functionIFNS_3anyEvEES4_E6invokeERNS1_15function_bufferE+00000028
% [ 65] 0x00007f004c795a85         /opt/MATLAB/R2019a/bin/glnxa64/libmwiqm.so+00768645 _ZN3iqm18PackagedTaskPlugin7executeEP15inWorkSpace_tag+00000437
% [ 66] 0x00007f004c356b35         /opt/MATLAB/R2019a/bin/glnxa64/libmwmcr.so+00944949
% [ 67] 0x00007f004c77cf2d         /opt/MATLAB/R2019a/bin/glnxa64/libmwiqm.so+00667437
% [ 68] 0x00007f004c75feba         /opt/MATLAB/R2019a/bin/glnxa64/libmwiqm.so+00548538
% [ 69] 0x00007f004c760b2f         /opt/MATLAB/R2019a/bin/glnxa64/libmwiqm.so+00551727
% [ 70] 0x00007f004c33de95         /opt/MATLAB/R2019a/bin/glnxa64/libmwmcr.so+00843413
% [ 71] 0x00007f004c33e4b3         /opt/MATLAB/R2019a/bin/glnxa64/libmwmcr.so+00844979
% [ 72] 0x00007f004c33ed24         /opt/MATLAB/R2019a/bin/glnxa64/libmwmcr.so+00847140
% [ 73] 0x00007f0052c13bdd /opt/MATLAB/R2019a/bin/glnxa64/libmwboost_thread.so.1.65.1+00080861
% [ 74] 0x00007f005346bac3                    /lib/x86_64-linux-gnu/libc.so.6+00608963
% [ 75] 0x00007f00534fd850                    /lib/x86_64-linux-gnu/libc.so.6+01206352
% [ 76] 0x0000000000000000                                   <unknown-module>+00000000
% 
% .
