%% Multigrid Deflation Method
clc; clear;

% n = 10;
% rng("default")
% A = rand(n);
% rhs = (1:n)';
A = load('./A_level2.mat').A;
n = size(A,1);
rhs = load('./rhs_level2.mat').x;
rhs = rhs(:,1);

a = A;
t=@(a,b)bicgstab(a,b,1e-1,100); % -> A^{-1}*x with a precision of 1e-1

k = [1, 2, 8, 16, 32];
res = {};
for i = k(:)'
[v, d] = eigs(@(x)t(a',t(a,x)), n, i, ...
    'largestimag', 'Tolerance',0.3, 'MaxIterations',30); 
v = orth(v);
           % t(a, x) => inv(A)x
           % t(a',t(a,x)) => inv(a') inv(A)x
% Hyper Param:
%   tol of eigs, tol of solver
%   # of eig vals/vecs

% If V are exact eigenvectors, then P=V*V'
% if V and L are eigenvectors and eigenvalues of A, then A*V=V*L,
% If A*V=V*L, and because V'*V=I, V'*A*V = V'*V*L = L
% P=A*V*inv(V'*A*V)*V' = V*L*inv(L)*V'=V*V'
% A^{-1}*b = A^{-1}*P*b + A^{-1}*(I-P)*b

% multigrid deflation P=A*V*inv(V'*A*V)*V'
inv_time = @(x) (v'*a*v) \ x;   % v'av -- coarse operator
P = @(x) a * (v * inv_time(v'*x));

% A^{-1}*b = A^{-1}*P*b + A^{-1}*(I-P)*b = V*inv(V'*A*V)*V'b + A^{-1}*(I-P)*b 
maxit =30;  % 10, 20 doesn't conv at 1e-1; 30 doesn't conv at 1e-2
tol = 1e-1; % try 1e-2 tol
Ainvb = @(A, b) v*inv_time(v'*b) + bicgstab(A, b - P(b), tol, maxit); 

[sol_x,~,~,iters,resvec] = bicgstab(A, rhs, 1e-5, 30, [], @(x)Ainvb(A,x));
res{end+1} = resvec;
res
end

out_tol = 1e-5; out_maxit = 30;
[sol_x1,~,~,iters1,resvec1] = bicgstab(A, rhs, out_tol, out_maxit, [], []);
% Ainvb = @(b) A \ P(b) + A \ (b - P(b));
% sol_x = Ainvb(A, rhs);
%%
clf; figure
for i=1:5
    semilogy((1:numel(res{i}))*maxit, res{i}); hold on
end

semilogy(resvec1)
legend("Ainvb, 1 eig", "Ainvb, 2 eig","Ainvb, 8 eig", ...
    "Ainvb, 16 eig", "Ainvb, 32 eig", "Unprecond.")

title(sprintf("Inner tol = %g, Inner maxit = %g, Out tol = %g, Out maxit = %g",tol, maxit, out_tol, out_maxit))
