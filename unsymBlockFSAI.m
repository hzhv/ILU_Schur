function [precond, FL, FU, D] = unsymBlockFSAI(A, n_blocks)
% Input:
%   n_blocks : # of blocks = np
% Output:
%   (x) precond 

    n = size(A, 1);    
    S = spones(A) | spones(A');     % Symm the nnz pattern
    
    block_size = floor(n / n_blocks);
    limits = [1:block_size:n, n+1]; % Blocks start idx
    
    FL = speye(n); FU = speye(n);

    for kb = 2:n_blocks % 1st block no dependencies, no need to calc the "interfaces"
        
        idx_start = limits(kb);
        idx_end   = limits(kb+1) - 1;
        curr_cols = idx_start:idx_end;
        
        M_ib = 1:(idx_start-1); % each r/c of Previous Blocks
        
        % Get FU and FL 
        for k = 1:length(curr_cols)
            idx = curr_cols(k);
            
            % get FU's idx col
            g_U = find(S(M_ib, idx)); % nnz indices for f (upper)
            g_L = g_U;                % nnz indices for f (left)
            A_sub = A(g_L, g_U);

            if ~isempty(g_U)
                % A_ib[gL, gU] * fU = -c  
                rhs   = -A(g_L, idx);
                
                f_val = A_sub \ rhs; % test
                
                FU(g_U, idx) = f_val;
            end
            
            % A(gU, gL)^T * fL = -r
            if ~isempty(g_L)
                A_sub_T = A_sub'; 
                rhs_T   = -A(idx, g_U)';
                
                f_val = A_sub_T \ rhs_T;
                
                FL(idx, g_L) = f_val'; 
            end
        end
    end
    
    M_approx = FL * A * FU;
    
    % Extract Block Diag B
    D_inv_ops = cell(n_blocks, 1);
    
    D = sparse(n, n);    
    for kb = 1:n_blocks
        idx = limits(kb):(limits(kb+1)-1);
        Block = M_approx(idx, idx);
        D(idx, idx) = Block;
        
        [L_blk, U_blk] = lu(Block); 
        D_inv_ops{kb}.L = L_blk;
        D_inv_ops{kb}.U = U_blk;
    end
    
    % M^{-1} = WU*WL = FU JU^{-1}JL^{-1} * FL = FU * D^{-1} * FL
    % z = FL * r -> y = D^(-1) * z -> x = FU * y
    precond = @(r) apply_solve(r, FL, FU, D_inv_ops, limits, n_blocks);
end

function x = apply_solve(r, FL, FU, D_inv_ops, limits, n_blocks)
    z = FL * r;
    
    % invert the bds (parallelizable)
    y = zeros(size(z));
    for kb = 1:n_blocks
        idx = limits(kb):(limits(kb+1)-1);
        % y(idx) = B_kb \ z(idx)
        ops = D_inv_ops{kb};
        block_rhs = ops.U \ (ops.L \ z(idx));
        y(idx) = block_rhs;
    end
    
    x = FU * y;
end