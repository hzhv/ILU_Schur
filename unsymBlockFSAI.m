function [precond, FL, FU, D] = unsymBlockFSAI(A, n_blocks)
% Input:
%   n_blocks : # of blocks = np
% Output:
%   (x) precond 

    n = size(A, 1);    
    S = spones(A) | spones(A');     % Symm the nnz pattern % TODO nnz inside our A's 64x64 blocks
    
    block_size = floor(n / n_blocks);
    limits = [1:block_size:n, n+1]; % Blocks start idx
    
    FL = speye(n); FU = speye(n);

    for kb = 2:n_blocks % 1st block no dependencies, no need to calc the "interfaces"
        
        idx_start = limits(kb);
        idx_end   = limits(kb+1) - 1;
        curr_cols = idx_start:idx_end;
        
        m_ib = 1:(idx_start-1);           % each r/c idx of Previous Blocks M_ib
        S_block = S(m_ib, curr_cols);
        g_U = m_ib(find(any(S_block,2))); % global row idx
        % g_L = m_ib(find(any(S(curr_cols, m_ib),2)));
        g_L = g_U;

        A_sub = A(g_U, g_L); % sparse
        % % A(gL,gU) * f = -A(gU, idx)
        rhs_U = -A(g_U, curr_cols);   
        F_block_U = A_sub \ rhs_U;       
        % mask_U = S(g_U, curr_cols);
        % F_block_U(~mask_U) = 0;
        FU(g_U, curr_cols) = F_block_U;
        
        % % A(gU, gL)^T * fL = -r'  ==>  fL' * A(gU, gL) = -r  
        rhs_L = -A(curr_cols, g_L)';
        F_block_L = (A_sub') \ rhs_L;
        FL(curr_cols, g_L) = F_block_L';
        
    end
    
    M_approx = FL * A * FU; 
    
    D = sparse(n, n);
    for kb = 1:n_blocks
        idx = limits(kb):(limits(kb+1)-1);
        Block = M_approx(idx, idx);
        D(idx, idx) = inv(Block);
    end
    
    precond = @(r) FU * (D *(FL * r));
end

% function x = apply_solve(r, FL, FU, D_inv_ops, limits, n_blocks)
%     z = FL * r;
    
%     % invert the bds (parallelizable)
%     y = zeros(size(z));
%     for kb = 1:n_blocks
%         idx = limits(kb):(limits(kb+1)-1);
%         % y(idx) = B_kb \ z(idx)
%         ops = D_inv_ops{kb};
%         block_rhs = ops.U \ (ops.L \ z(idx));
%         y(idx) = block_rhs;
%     end
    
%     x = FU * y;
% end