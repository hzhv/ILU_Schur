clc; clear; close all; reset(gpuDevice);
D = [8 8 8 16];
p = [0 0 0 0];
k_total = 3;
tol = 1e-6;

A = lap_kD_periodic(D,1); % eigen vector all the same % this on GPU
% figure; spy(A); title("Original A");
N = size(A,1);
% A = A + speye(N)*1e-7; % non-singular
%A = A.*(1+rand(N,N)/10) + speye(N)*1e-7;
rng(1); parallel.gpu.rng(1, 'Philox');
A = kron(A, (rand(1)));   % For denser "boxed" diag
B_perm = ones(1,1);

N_new = size(A, 1);
A = A + speye(N_new)*1e-7; % non-singular

Ag = gpuArray(A); % GPU
% figure; spy(Ag); title("After Kron(A, rand(48))")

b = ones(N_new, 1);  % RHS: all-ones vector
bg = randn(N_new, 1);
% bg = randn(N_new, 1, "gpuArray");

% === For GMRES === %
maxit = 1000;
restart = 100;

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
    [LM_ee, UM_ee] = lu(M_ee); % no zero pivot I guess?
    % AK(K^{-1}x)=y (Right Precondtioning) -> AKt=y -> x=Kt
    LM_ee_g = gpuArray(LM_ee); UM_ee_g = gpuArray(UM_ee);
    M_handle = @(xg) UM_ee_g\(LM_ee_g\xg);
    
    M_handle_old = @(xg) M_ee_g\xg;
    tic;
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