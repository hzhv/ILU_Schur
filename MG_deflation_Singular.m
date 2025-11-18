function [sol_x, res_inner, inner_iter_vec, resvec_outer, outer_iters] = MG_deflation_Singular( ...
    A, rhs, u, v, ...
    tol_inner, maxit_inner, ...
    tol_outer, maxit_outer, ...
    precond, M_smo, solver)

if nargin < 10 || isempty(solver) || strcmpi(solver, 'bicgstab')
    solver = @(A,b,tol,maxit,M1,M2,x0) bicgstab(A,b,tol,maxit,[],M2,x0);
else
    solver = @(A,b,tol,maxit,M1,M2,x0) min_res_sd(A,b,tol,maxit,[],M2,x0);
end

res_inner = {};
inner_iter_vec = [];

inv_time = @(x) (v'*A*u) \ x;
P = @(x) A * (u*inv_time(v'*x));

total_inner_iters = 0;   

if precond == 0      % unprec
    M2 = [];
elseif precond == 1  % deflation w/ unprec. smoother
    M2 = @(x) Ainvb_with_count(x, []);
elseif precond == 2  % M2 = ilu0 
    M2 = M_smo;
elseif precond == 3  % deflation w/ ilu(0) smoother
    M2 = @(x) Ainvb_with_count(x, M_smo);
elseif precond == 4  
    M2 = M_smo;      % M2 = other preconds
elseif precond == 5  % deflation w/ other precond smoothers
    M2 = @(x) Ainvb_with_count(x, M_smo);
end

[sol_x, ~, relres, outer_iters, resvec_outer] = solver(A, rhs, tol_outer, maxit_outer, [], M2, []); % A M^{-1} y = rhs
                                                                                                    % x = M^{-1}y
total_iters = outer_iters + total_inner_iters;

fprintf('Outer iters: %d\n', outer_iters);
fprintf('Last Relative Residual: %d\n', relres)
fprintf('Total iters: %d\n', total_iters);

    function y = Ainvb_with_count(b_in, M2) 
        coarse = u * inv_time(v'*b_in);   % course, y_coarse = A^{-1} P b

        rtilde = b_in - P(b_in);          % smoother, A z = (I - P) b

        % % inv(A)b = inv(A)Pb + inv(A)(I-P)b
        [y, flag, ~, it_in, resvec_inner] = gmres(A, b_in, maxit_inner, tol_inner, 1, [], M2, coarse); 
        gmres_iter = (it_in(1)-1)*maxit_inner + it_in(2);
              
        total_inner_iters = total_inner_iters + gmres_iter;       

        res_inner{end+1} = resvec_inner;
        inner_iter_vec(end+1)= gmres_iter;
    end

end