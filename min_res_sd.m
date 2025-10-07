function [x, flag, relres, iters, resvec] = min_res_sd(a, b, tol, maxit, M1, M2)
iters = 0;
n = size(a,1);
x = zeros(n,1);
r = b - a*x;
resvec = [norm(r)];

if nargin < 6 || isempty(M2)
    M2 = @(v) v; 
end

while norm(r)/norm(b) > tol
    kr = M2(r);     % kr = precond * r
    p = a*kr;
    alpha = (p' * r) / (p' * p);
    x = x + kr*alpha;
    % r = r - p * alpha;
    r = b - a*x;  % less numerical errors
    resvec = [resvec; norm(r)];
    iters = iters + 1;
    if iters >= maxit
        fprintf("Not converged to %g, reach the maxit %g\n", tol, maxit);
        fprintf("Last relres = %d\n\n", norm(r)/norm(b));
        flag = 1;
        break;
    end
end
flag = 0;
relres = norm(r)/norm(b);
end
