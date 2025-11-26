function z = dd_inv(dd, v, tol, maxit, precond)
    if nargin < 5, precond = []; end
    
    [z, flag, ~, iters] = minres(dd, v, ...
        tol, maxit,[], precond);
    if flag == 0,fprintf('Solving dd*z = v with %g iters\n', iters); end
end