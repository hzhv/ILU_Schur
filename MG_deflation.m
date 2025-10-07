function [sol_x, res_inner, inner_iter_vec, resvec_outer, total_iters] = MG_deflation(A, rhs, v, tol_inner, maxit_inner, ...
    tol_outer, maxit_outer, precond, solver)

if nargin < 8 || isempty(solver) || strcmpi(solver, 'bicgstab')
    solver = @(A,b,tol,maxit,M1,M2) bicgstab(A,b,tol,maxit,[],M2);
else
    solver = @(A,b,tol,maxit,M1,M2) min_res_sd(A,b,tol,maxit,[],M2);
end

res_inner = [];
inner_iter_vec = [];

inv_time = @(x) (v'*A*v) \ x; % coarse operator = (V' A V)^{-1} x
P        = @(x) A * (v * inv_time(v'*x));

% Counters
total_inner_iters = 0;   
precond_calls = 0;   % preconditioner calls

if precond == 0
    M2 = [];
else 
    M2 = @(x) Ainvb_with_count(x);
end

[sol_x, ~, relres, outer_iters, resvec_outer] = solver(A, rhs, tol_outer, maxit_outer, [], M2);

total_iters = outer_iters + total_inner_iters;

fprintf('Outer iters: %d\n', outer_iters);
fprintf('Last Relative Residual: %d\n', relres)
fprintf('Total iters: %d\n', total_iters);

    function y = Ainvb_with_count(b_in)    % Inner
        precond_calls = precond_calls + 1; % count preconditioner calls

        coarse = v * inv_time(v' * b_in);  % course, no iters

        rtilde = b_in - P(b_in); % smoother A z = (I - P) b

        [z, ~, ~, it_in, resvec_inner] = solver(A, rtilde, tol_inner, maxit_inner, [], []);

        total_inner_iters = total_inner_iters + it_in;
        
        res_inner = [res_inner; resvec_inner];
        inner_iter_vec = [inner_iter_vec;it_in];
        
        y = coarse + z;
    end   
end