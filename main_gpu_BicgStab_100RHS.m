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
bar_width = 50;                % ProgressBar width
last_msg_length = 0;           % record last print length
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
% b = load('rhs.mat').b;
b = load('rhs_level2.mat').x; % 32768x10
num_rhs = size(b, 2);
% b = (1:N_new)';
% b = rand(N_new,10);
bg = gpuArray(b);

% x0 = b;
%% "Pure BiCGstab Iterations"
all_iterations = zeros(1, num_rhs);
all_relres = zeros(1, num_rhs);
tic;
fprintf('Pure iterative results w/o any preconditioner:\n');
for bdx = 1:num_rhs
    [xg, flag, relres, iter, resvec_pure] = ... % 
        bicgstabl(Ag, bg(:,bdx), tol, maxit, [], []);
    total_iter_pure = iter;
    relres_true = norm(bg(:,bdx) - Ag*xg)/norm(bg(:,bdx));
    all_iterations(bdx) = total_iter_pure; 
    all_relres(bdx) = relres_true;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% ProgressBar %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    progress = bdx / num_rhs;
    completed_width = round(progress * bar_width);
    remaining_width = bar_width - completed_width;
     % ProgressBar String
    progress_bar = ['[' repmat('=', 1, completed_width) repmat(' ', 1, remaining_width) ']'];
    msg = sprintf('bicgstabl Start... %s %.1f%% (%d/%d)\n', progress_bar, progress * 100, bdx, num_rhs);
   
    fprintf(repmat('\b', 1, last_msg_length));
    fprintf('%s', msg);
    last_msg_length = length(msg); 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
tp = toc;

ave_iter_pure = mean(all_iterations);
fprintf('  The average exact residual norm = %d\n', mean(all_relres));
fprintf('  bicgstabl %f iterations. in average of %g\n', ave_iter_pure, num_rhs);
fprintf('  bicgstabl cost %f sec in average of %g\n', tp/num_rhs, num_rhs);
%% Use ILU(0) Preconditioner Only
all_iterations_ilu0 = zeros(1, num_rhs);
all_relres_ilu0 = zeros(1, num_rhs);
fprintf('ilu(0) natural, no multi-reordering:\n');

setup.type    = 'nofill';
setup.droptol = 0;  
[L, U] = ilu(A, setup);        % Matlab can't do ILU on GPU
Lg = gpuArray(L); Ug = gpuArray(U);

tic;
for bdx = 1:num_rhs
    M_handle = @(x) Ug\(Lg\x); % Handle in loop
    [x_perm_gpu, flag, relres, iter, resvec_ilu0] = ...
        bicgstabl(Ag, bg(:, bdx), tol, maxit, [], M_handle);
    
    total_iter = iter;
    relres_true = norm(bg(:,bdx) - Ag*x_perm_gpu)/norm(bg(:,bdx));
    all_iterations_ilu0(bdx) = total_iter; 
    all_relres_ilu0(bdx) = relres_true;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% ProgressBar %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    progress = bdx / num_rhs;
    completed_width = round(progress * bar_width);
    remaining_width = bar_width - completed_width;
     % ProgressBar String
    progress_bar = ['[' repmat('=', 1, completed_width) repmat(' ', 1, remaining_width) ']'];
    msg = sprintf('bicgstabl Start... %s %.1f%% (%d/%d)\n', progress_bar, progress * 100, bdx, num_rhs);
   
    fprintf(repmat('\b', 1, last_msg_length));
    fprintf('%s', msg);
    last_msg_length = length(msg); 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
t_ilu = toc;

ave_iter_ilu0 = mean(all_iterations_ilu0);
fprintf('  The average exact residual norm = %d\n', mean(all_relres_ilu0));
fprintf('  bicgstabl %f iterations. in average of %g\n', mean(all_iterations_ilu0), num_rhs);
fprintf('  bicgstabl cost %f sec in average of %g\n', t_ilu/num_rhs, num_rhs);
%% Use muticoloring only without Even-Odd ordering

fprintf('Only multi-coloring w/o Even-Odd:\n');
all_iterations_noEO = zeros(k_total, num_rhs);
all_relres_noEO = zeros(k_total, num_rhs);
all_resvecs = cell(1, num_rhs);
evenCs_noEO = zeros(1, k_total);

for k = 1:k_total
    [Colors, ncolor] = displacement_even_odd_coloring_nD_lattice(D, k, [0 0 0 0]);
    [isOK, evenCs, badCs, loc] = check_eo_compatibility_and_return_even(Colors, D, N_new);
    evenCs_noEO(k) = length(unique(evenCs));
    if ~isOK
        fprintf('  Not compatible. Conflicting colors: '); fprintf('%d ', badCs); fprintf('\n');
        disp(loc{1});
        error('Aborted.')
    else
        fprintf('  OK: coloring is even/odd compatible.\n\n');
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
    
    setup.type    = 'nofill';
    setup.droptol = 0;  
    [L, U] = ilu(A_perm, setup); % Matlab can't do ILU on GPU
    Lg = gpuArray(L); Ug = gpuArray(U);
    
    tic;
    for bdx = 1:num_rhs
        % if k == 3 && bdx == 39
        % keyboard;
        % end
        b_perm_gpu = bg(perm, bdx);
        M_handle = @(x) Ug\(Lg\x);
      
        [x_perm_gpu, flag, relres, iter, resvec_noEO] = ...
            bicgstabl(A_perm_gpu, b_perm_gpu, tol, maxit, [], M_handle);
        
        all_iterations_noEO(k, bdx) = iter;
        all_relres_noEO(k, bdx) = norm(b_perm_gpu - A_perm_gpu*x_perm_gpu)/norm(b_perm_gpu);
        all_resvecs{bdx} = resvec_noEO;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%% ProgressBar %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        progress = bdx / num_rhs;
        completed_width = round(progress * bar_width);
        remaining_width = bar_width - completed_width;
         % ProgressBar String
        progress_bar = ['[' repmat('=', 1, completed_width) repmat(' ', 1, remaining_width) ']'];
        msg = sprintf('bicgstabl Start... %s %.1f%% (%d/%d)', progress_bar, progress * 100, bdx, num_rhs);

        fprintf(repmat('\b', 1, last_msg_length));
        fprintf('%s', msg);
        last_msg_length = length(msg); 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    t_noEO = toc;
    fprintf('\n'); 
    
    max_iter = max(all_iterations_noEO(k,:));
    resvec_matrix = nan(max_iter, num_rhs);
    for i = 1:num_rhs
        current_resvec = all_resvecs{i};
        len = length(current_resvec);
        resvec_matrix(1:len, i) = current_resvec(:); 
    end
    average_resvec = mean(resvec_matrix, 2, 'omitnan');
    
    res_noEO{k} = average_resvec;
    iters(k) = mean(all_iterations_noEO(k,:));
    nColors_noEO(k) = nColors;

    if flag == 0
        fprintf('  When k = %d,', k);
        fprintf('  total colors = %d\n', nColors_noEO(k));
        fprintf('  The average exact residual norm = %d\n', mean(all_relres_noEO(k,:)));
        fprintf('  bicgstabl %f iterations. in average of %g\n', iters(k), num_rhs);
        fprintf('  bicgstabl cost %f sec in average of %g\n', t_noEO/num_rhs);
    else
        fprintf('  bicgstabl failed to converge (flag = %d). Relative residual = %e.\n', flag, relres);
    end
    
end
%% Multi-reordering with Even-Odd Reordering
reset(gpuDevice);
fprintf('Check the EO and Coloring compatibility first...:\n')
all_iterations_EO = zeros(k_total, num_rhs);
all_relres_EO = zeros(k_total, num_rhs);
all_resvecs = cell(1, num_rhs);
evenCs_EO = zeros(1, k_total);

for k = 1:k_total
    [Colors, ncolor] = displacement_even_odd_coloring_nD_lattice(D, k, [0 0 0 0]);
    [isOK, evenCs, badCs, loc] = check_eo_compatibility_and_return_even(Colors, D, N_new);
    evenCs_EO(k) = length(unique(evenCs));
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

    setup.type    = 'nofill';
    setup.droptol = 0;  
    [L, U] = ilu(A_perm, setup);
    Lg = gpuArray(L); Ug = gpuArray(U);

    tic;
    for bdx = 1:num_rhs
        b_perm_gpu = b(perm, bdx);
        
        M_handle = @(x) Ug\(Lg\x);
        % x0 = b_perm;
        
        [x_perm_gpu, flag, relres, iter, resvec_EO] = ...
            bicgstabl(A_perm_gpu, b_perm_gpu, tol, maxit, [], M_handle); % M1=Lg? M2=Ug?

        all_iterations_EO(k, bdx) = iter;
        all_relres_EO(k, bdx) = norm(b_perm_gpu - A_perm_gpu*x_perm_gpu)/norm(b_perm_gpu);
        all_resvecs{bdx} = resvec_EO;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%% ProgressBar %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        progress = bdx / num_rhs;
        completed_width = round(progress * bar_width);
        remaining_width = bar_width - completed_width;
         % ProgressBar String
        progress_bar = ['[' repmat('=', 1, completed_width) repmat(' ', 1, remaining_width) ']'];
        msg = sprintf('bicgstabl Start... %s %.1f%% (%d/%d)', progress_bar, progress * 100, bdx, num_rhs);

        fprintf(repmat('\b', 1, last_msg_length));
        fprintf('%s', msg);
        last_msg_length = length(msg); 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    t_eo = toc;
    fprintf('\n'); 
    
    max_iter = max(all_iterations_EO(k,:));
    resvec_matrix = nan(max_iter, num_rhs);
    for i = 1:num_rhs
        current_resvec = all_resvecs{i};
        len = length(current_resvec);
        resvec_matrix(1:len, i) = current_resvec(:); 
    end
    average_resvec = mean(resvec_matrix, 2, 'omitnan');
    
    iters_eo(k) = mean(all_iterations_EO(k,:));
    res_eo{k} = average_resvec;
    nColors_EO(k) = nColors;

    if flag == 0
        fprintf('  When k = %d,', k);
        fprintf('  total colors = %d\n', nColors_EO(k));
        fprintf('  The average exact residual norm = %d\n', mean(all_relres_EO(k,:)));
        fprintf('  bicgstabl %f iterations. in average of %g\n', iters_eo(k), num_rhs);
        fprintf('  bicgstabl cost %f sec in average of %g\n', t_eo/num_rhs);
    end
end
% Undo permutation to original order
% invperm(perm) = 1:length(perm);
% x = x_perm(invperm);

%% Partial ILU(0)(A) with Schur Complement

fprintf('bicgstabl Schur Complement on GPU:\n')
all_iterations_even = zeros(k_total, num_rhs);
all_relres_even = zeros(k_total, num_rhs);
all_resvecs = cell(1, num_rhs);
evenCs_Schur = zeros(1, k_total);

for k = 1:k_total
    [Colors, ncolor] = displacement_even_odd_coloring_nD_lattice(D, k, [0 0 0 0]);
    [isOK, evenCs, badCs, loc] = check_eo_compatibility_and_return_even(Colors, D, N_new);
    evenCs_Schur(k) = length(unique(evenCs));
    if ~isOK
        fprintf('  Not compatible. Conflicting colors: '); fprintf('%d ', badCs); fprintf('\n');
        disp(loc{1});
        error('Aborted.')
    else
        fprintf('  OK: coloring is even/odd compatible.\n');
    end
end

iters_schur = zeros(1, k_total);
res_schur = cell(1, k_total);
n = N_new;
nColors_Schur = zeros(1, k_total);

for k = 1:k_total 
    [colors, nColors] = displacement_even_odd_coloring_nD_lattice(D, k, p);
    colors = kron(colors, B_perm);
    [~, perm] = sort(colors);
    A_perm = A(perm, perm);
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
    
    % === Preconditioned Handle (Left Preconditioning) === %
    setup.type    = 'nofill';
    setup.droptol = 0;  
    [L, U] = ilu(A_perm, setup);      % on CPU
    M = L*U; 
    M_ee_old = M(1:n/2, 1:n/2);       % "Cut" on CPU
    q_M = colamd(M_ee_old);
    M_ee = M_ee_old(:, q_M);

    % === AK(K^{-1}x)=y (Right Precondtioning) -> AKt=y -> x=Kt ===
    % M_ee, M_oo are pure block diagonal matrices, no zero pivot
    % [LM_ee, UM_ee] = lu(M_ee);  % intro fill-ins but acc bicgstabl
    % M_handle = @(xg) gpuArray(UM_ee\(LM_ee\gather(xg))); % acc 100% in sec  
    
    tic;
    for bdx = 1:num_rhs
        b_perm_gpu = bg(perm, bdx);
        % Eliminate odd -> bicgstabl solve for even
        bp_e_gpu = b_perm_gpu(1:n/2);
        bp_o_gpu = b_perm_gpu(n/2+1 : end);
        % rhs_o = bp_o - ap_oe * (ap_ee \ bp_e);
        rhs_e_gpu = bp_e_gpu - ap_eo * solve_oo_gpu(bp_o_gpu);

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
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [x_even_gpu, flag, relres_even, iter, resvec_even] = ...
            bicgstabl(s_ee_gpu, rhs_e_gpu, tol, maxit, [], M_handle_old);
        x_odd = ap_oo \ (bp_o_gpu - ap_oe * x_even_gpu);

        all_iterations_even(k, bdx) = iter;
        all_relres_even(k, bdx) = norm(rhs_e_gpu - s_ee_gpu(x_even_gpu))/norm(b_perm_gpu);
        all_resvecs{bdx} = resvec_even;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%% ProgressBar %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        progress = bdx / num_rhs;
        completed_width = round(progress * bar_width);
        remaining_width = bar_width - completed_width;
         % ProgressBar String
        progress_bar = ['[' repmat('=', 1, completed_width) repmat(' ', 1, remaining_width) ']'];
        msg = sprintf('bicgstabl Start... %s %.1f%% (%d/%d)', progress_bar, progress * 100, bdx, num_rhs);

        fprintf(repmat('\b', 1, last_msg_length));
        fprintf('%s', msg);
        last_msg_length = length(msg); 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    t_imp = toc;
    fprintf('\n'); 
    
    max_iter = max(all_iterations_even(k,:));
    resvec_matrix = nan(max_iter, num_rhs);
    for i = 1:num_rhs
        current_resvec = all_resvecs{i};
        len = length(current_resvec);
        resvec_matrix(1:len, i) = current_resvec(:); 
    end
    average_resvec = mean(resvec_matrix, 2, 'omitnan');
    
    iters_schur(k) = mean(all_iterations_EO(k,:));
    res_schur{k} = average_resvec;
    nColors_Schur(k) = nColors;

    if flag == 0
        fprintf('  When k = %d,', k);
        fprintf('  total colors = %d\n', nColors);
        fprintf('  The average exact residual norm = %d\n', mean(all_relres_EO(k, :)));
        fprintf('  bicgstabl %f iterations. in average of %g\n', iters_schur(k), num_rhs);
        fprintf('  bicgstabl cost %f sec in average of %g\n', t_imp/num_rhs);
    end

    % === Combine x_even and x_odd ===
end
%% bicgstabl Schur Complement w/o preconditioner

fprintf('bicgstabl Schur Complement no precondtioner:\n')
all_iters_schur_no_prec = zeros(k_total, num_rhs);
all_relres_schur_no_prec = zeros(k_total, num_rhs);
all_resvecs = cell(1, num_rhs);
evenCs_schur_no_prec = zeros(1, k_total);

for k = 1:k_total
    [Colors, ncolor] = displacement_even_odd_coloring_nD_lattice(D, k, [0 0 0 0]);
    [isOK, evenCs, badCs, loc] = check_eo_compatibility_and_return_even(Colors, D, N_new);
    evenCs_schur_no_prec(k) = length(unique(evenCs));
    if ~isOK
        fprintf('  Not compatible. Conflicting colors: '); fprintf('%d ', badCs); fprintf('\n');
        disp(loc{1});
        error('Aborted.')
    else
        fprintf('  OK: coloring is even/odd compatible.\n');
    end
end

iters_schur_no_prec = zeros(1, k_total);
resvec_schur_no_prec = cell(1, k_total);
n = N_new;
nColors_schur_no_prec = zeros(1, k_total);

for k = 1:k_total 
    [colors, nColors] = displacement_even_odd_coloring_nD_lattice(D, k, p);
    colors = kron(colors, B_perm);
    [~, perm] = sort(colors);
    A_perm = A(perm, perm);
    ap_ee = A_perm(1:n/2,1:n/2);          ap_ee = gpuArray(ap_ee);
    ap_eo = A_perm(1:n/2, n/2+1:end);     ap_eo = gpuArray(ap_eo);
    ap_oe = A_perm(n/2+1:end, 1:n/2);     ap_oe = gpuArray(ap_oe);

    ap_oo = A_perm(n/2+1:end, n/2+1:end); ap_oo = gpuArray(ap_oo);
 
    solve_oo_gpu = @(xg) ap_oo\xg; % normal

    s_ee_gpu = @(xg)...
       ap_ee*xg - ap_eo * solve_oo_gpu(ap_oe*xg);

    tic;
    for bdx = 1:num_rhs
        b_perm_gpu = bg(perm, bdx);
        % Eliminate odd -> bicgstabl solve for even
        bp_e_gpu = b_perm_gpu(1:n/2);
        bp_o_gpu = b_perm_gpu(n/2+1 : end);
        % rhs_o = bp_o - ap_oe * (ap_ee \ bp_e);
        rhs_e_gpu = bp_e_gpu - ap_eo * solve_oo_gpu(bp_o_gpu);

        [x_schur_no_prec_gpu, flag, relres_even, iter, resvec_even] = ...
            bicgstabl(s_ee_gpu, rhs_e_gpu, tol, maxit, [], []);
        x_odd = ap_oo \ (bp_o_gpu - ap_oe * x_schur_no_prec_gpu);

        all_iters_schur_no_prec(k, bdx) = iter;
        all_relres_schur_no_prec(k, bdx) = norm(rhs_e_gpu - s_ee_gpu(x_schur_no_prec_gpu))/norm(b_perm_gpu);
        all_resvecs{bdx} = resvec_even;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%% ProgressBar %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        progress = bdx / num_rhs;
        completed_width = round(progress * bar_width);
        remaining_width = bar_width - completed_width;
         % ProgressBar String
        progress_bar = ['[' repmat('=', 1, completed_width) repmat(' ', 1, remaining_width) ']'];
        msg = sprintf('bicgstabl Start... %s %.1f%% (%d/%d)', progress_bar, progress * 100, bdx, num_rhs);

        fprintf(repmat('\b', 1, last_msg_length));
        fprintf('%s', msg);
        last_msg_length = length(msg); 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    t_schur_no_prec = toc;
    fprintf('\n'); 
    
    max_iter = max(all_iters_schur_no_prec(k,:));
    resvec_matrix = nan(max_iter, num_rhs);
    for i = 1:num_rhs
        current_resvec = all_resvecs{i};
        len = length(current_resvec);
        resvec_matrix(1:len, i) = current_resvec(:); 
    end
    average_resvec = mean(resvec_matrix, 2, 'omitnan');
    
    iters_schur_no_prec(k) = mean(all_iters_schur_no_prec(k,:));
    resvec_schur_no_prec{k} = average_resvec;
    nColors_schur_no_prec(k) = nColors;

    if flag == 0
        fprintf('  When k = %d,', k);
        fprintf('  total colors = %d\n', nColors);
        fprintf('  The average exact residual norm = %d\n', mean(all_relres_schur_no_prec(k, :)));
        fprintf('  bicgstabl %f iterations. in average of %g\n', iters_schur_no_prec(k), num_rhs);
        fprintf('  bicgstabl cost %f sec in average of %g\n', t_schur_no_prec/num_rhs);
    end

    % === Combine x_even and x_odd ===
end
%% Plot all Convergence
figure;
for k = 1:k_total
    semilogy(res_schur{k},'-', 'LineWidth', 1.2, 'DisplayName', sprintf('Schur, k = %d', k));
    hold on;
    semilogy(resvec_schur_no_prec{k}, 'LineWidth', 2, 'DisplayName', sprintf('Schur, no prep., k = %d', k));
    hold on;
end
xlabel('Iteration');
ylabel('Residual Norm');
% yline(tol,'r--','DisplayName', sprintf('Tol'));
title('Schur Comp. bicgstabl Residual Convergence with EO');
legend show;
grid on;

figure; clf;
for k = 1:k_total
    semilogy(res_noEO{k},'--' ,'LineWidth', 1.2, 'DisplayName', sprintf('w/o EO, k = %d', k));
    hold on;
    semilogy(res_eo{k},'-', 'LineWidth', 1.2, 'DisplayName', sprintf('w/ EO, k = %d', k));
    hold on;
end
xlabel('Iteration');
ylabel('Residual Norm');
% yline(tol,'r--','DisplayName', sprintf('Tol'));
title('Schur Comp. bicgstabl Residual Convergence with EO');
legend show;
grid on;

figure; clf;
semilogy(resvec_pure,'-', 'LineWidth', 1.2, 'DisplayName', sprintf('Pure bicgstabl'));
hold on;
semilogy(resvec_ilu0,'-', 'LineWidth', 1.2, 'DisplayName', sprintf('ILU(0) Natural'));

xlabel('Iteration');
ylabel('Residual Norm');
% yline(tol,'r--','DisplayName', sprintf('Tol'));
title('Schur Comp. bicgstabl Residual Convergence with EO');
legend show;
grid on;
%%
figure; subplot(1,2,1);
X = 1:k_total;
Y = [iters(:), iters_eo(:), iters_schur(:), iters_schur_no_prec(:)];
total_colors = [nColors_noEO(:), nColors_EO(:), nColors_Schur(:), nColors_Schur(:)];
even_colors = [evenCs_noEO(:), evenCs_EO(:), evenCs_Schur(:), evenCs_schur_no_prec(:)];

bar(X, Y, 'grouped'); hold on;

yline(ave_iter_pure, '--r', 'Pure bicgstabl', 'LineWidth', 2);
yline(ave_iter_ilu0, '--b', 'ILU(0) Natural', 'LineWidth', 2);

xlabel('k');
ylabel('bicgstabl Iterations');
legend('Without EO','With EO','Schur Comp.', 'Schur Comp. w/o ILU(0)','Pure bicgstabl','ILU(0) Natural','Location','northwest');
title('bicgstabl Iterations vs. k');
grid on;

subplot(1,2,2);
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