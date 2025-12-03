%%
% clc;clear;
% rng(0);
% A = sprandn(100,100,0.01); A = A' * A + 1e-3 * speye(100);
% condA = condest(A);

% A = load("A_level2.mat").A; condA = 2.2393e+04;
% [U,S,V] = getSingularTrip(A, 100, condA/0.1, 1000);
%% 
function [U,S,V] = getSingularTrip(A, k, tol, maxit)  
    [l, u] = ilu(A, struct('type','nofill'));
    % U'L' = A'
    t =@(A,b) tfqmr(A,  b, tol*tol*0.3, 1000, [], @(x) u\(l\x));
    tp=@(A,b) tfqmr(A', b, tol*tol*0.3, 1000, [], @(x) l'\(u'\x));

    addpath('./primme/Matlab/');
    % [U,S,V] = svds(@svds_matvect, size(A), k, 'largest', ...                                                                          t(A, b), size(A), k, 'largest', ...
    %      'Tolerance', tol, 'MaxIterations',maxit);
    opts = struct('tol', tol,'maxit',maxit, ...
        'reportLevel',2, 'method','primme_svds_normalequations', ...
        'primme',struct('method','DEFAULT_MIN_MATVECS'));

    Afun = @(x, flag) svds_matvect(x, flag, A);
    [U, S, V] = primme_svds(Afun, size(A,1), size(A,2), k, 'S', opts);

    SCell = {U, S, V};
    save("singularTripL2_Schur.mat", "SCell");
    

    function y = svds_matvect(x, flag, A)
        if nargin < 2 || strcmp(flag, "notransp")
            % y = t(A,x);
            y = A*x;
        else
            % y = tp(A',x);
            y = A'*x;
        end
    end
end
