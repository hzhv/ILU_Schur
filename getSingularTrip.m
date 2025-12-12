%%
% clc;clear;
% rng(0);
% A = sprandn(100,100,0.01); A = A' * A + 1e-3 * speye(100);
% condA = condest(A);

% A = load("A_level2.mat").A; condA = 2.2393e+04;
% [U,S,V] = getSingularTrip(A, 100, condA/0.1, 1000);
%% 
function [U,S,V] = getSingularTrip(A, k, tol, maxit)  
    % Notice that this script is currently getting the largest k  S.trips of A^{-1},
    %  which equivalent to the smallest k  S.trips of A
    % U'L' = A'
    disp('Computing ILU preconditioner...');
    [l, u] = ilu(A, struct('type','nofill')); disp('Done.');
    lt = l'; ut = u';
    t =@(A,b) tfqmr(A,  b, tol*tol*0.3, 10000, [], @(x) u\(l\x));
    tp=@(A,b) tfqmr(A', b, tol*tol*0.3, 10000, [], @(x) lt\(ut\x));

    addpath('./primme/Matlab/');
    opts = struct('tol', tol,'maxit',maxit, ...
        'reportLevel',2, 'method','primme_svds_normalequations', ...
        'primme',struct('method','DEFAULT_MIN_MATVECS'));

    Afun = @(x, flag) svds_matvect(A, x, flag);
    [U, S, V] = primme_svds(Afun, size(A,1), size(A,2), k, 'L', opts);

    SCell = {U, S, V};
    save("singularTripL1_Schur.mat", "SCell");
    

    function y = svds_matvect(A, x, flag)
        y = zeros(size(x));
        
        if nargin < 2 || strcmp(flag, "notransp")
            for i = 1:size(x, 2)
                % y = A*x;
                [y(:, i), flg, relres] = t(A,x(:, i));
                if flg ~= 0, warning("tfqmr did not converged! Flag: %d, RelRes: %e", flg, relres); end
            end
        else
            for i = 1:size(x, 2)
                % y = A'*x;
                [y(:, i), flg, relres] = tp(A,x(:, i));
                if flg ~= 0, warning("tfqmr did not converged! Flag: %d, RelRes: %e", flg, relres); end        
            end
        end

        
    end
end
