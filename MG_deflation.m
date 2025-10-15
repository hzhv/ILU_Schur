function [sol_x, res_inner, inner_iter_vec, relresvec_outer, outer_iters] = MG_deflation( ...
    A, rhs, v, ...
    tol_inner, maxit_inner, ...
    tol_outer, maxit_outer, ...
    precond, M_smo, solver)

if nargin < 10 || isempty(solver) || strcmpi(solver, 'bicgstab')
    solver = @(A,b,tol,maxit,M1,M2,x0) bicgstab(A,b,tol,maxit,[],M2,x0);
else
    solver = @(A,b,tol,maxit,M1,M2,x0) min_res_sd(A,b,tol,maxit,[],M2,x0);
end


res_inner = [];
inner_iter_vec = [];

inv_time = @(x) (v'*A*v) \ x;    % coarse operator = inv(V' A V) x
P = @(x) A * (v*inv_time(v'*x));

% Counters
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
    M2 = M_smo;      % M2 = bj
elseif precond == 5  % deflation w/ bj smoother
    M2 = @(x) Ainvb_with_count(x, M_smo);
end

[sol_x, ~, relres, outer_iters, resvec_outer] = solver(A, rhs, tol_outer, maxit_outer, [], M2, []); % A M^{-1} y = rhs
                                                                                                    % x = M^{-1}y
relresvec_outer = resvec_outer/norm(rhs);

total_iters = outer_iters + total_inner_iters;

fprintf('Outer iters: %d\n', outer_iters);
fprintf('Last Relative Residual: %d\n', relres)
fprintf('Total iters: %d\n', total_iters);

    function y = Ainvb_with_count(b_in, M2) % Inner

        coarse = v * inv_time(v'*b_in);   % course, y_coarse = A^{-1} P b

        rtilde = b_in - P(b_in);          % smoother, A z = (I - P) b
   
        % % inv(A)b = inv(A)Pb + inv(A)(I-P)b
        [y, ~, ~, it_in, resvec_inner] = solver(A, b_in, tol_inner, maxit_inner, [], M2, coarse); 
        % [z, ~, ~, it_in, resvec_inner] = solver(A, rtilde, tol_inner, maxit_inner, [], M2, zeros(size(b_in))); % Friday Bug
        total_inner_iters = total_inner_iters + it_in;       
        % y = coarse + z;
        
        res_inner = [res_inner; resvec_inner];
        inner_iter_vec = [inner_iter_vec;it_in];
    end
   
end