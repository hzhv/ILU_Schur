function geo_mean = test_mgd_singular(...
    A, B, u, v, ...
    tol_inner, maxit_inner, ...
    tol_outer, maxit_outer, ...
    precond, M_smo, solver)

m = size(B, 2);
Xmax_each = zeros(m,1);         % gather total inner iteratons per rhs
interp_ln_resvecs = cell(m,1);  % interp on 1:Xmax, gather'em all

for j = 1:m
        rhs = B(:, j);
        [~, ~, inner_iter_vec, resvec_outer, ~] = MG_deflation_Singular( ...
            A, rhs, u, v, ...
            tol_inner, maxit_inner, ...
            tol_outer, maxit_outer, ...
            precond, M_smo, solver);
    resvec_outer = resvec_outer/norm(rhs);

    if precond == 0 || precond == 2 || precond == 4
        X = (1:numel(resvec_outer)); 
    else
        X = [1; cumsum(inner_iter_vec(:))];
    end
    assert(numel(X) == numel(resvec_outer), ...
        'inner_iter_vec size must match resvec_outer(2:end)');
    [Xu, ia] = unique(X, 'stable');  % X_unique
    resvec_outer = resvec_outer(ia);

    Xq = (1:max(Xu));
    ln_resvec_outer = log(resvec_outer);
    ln_resvec_q = interp1(Xu, ln_resvec_outer, Xq, 'linear');
    interp_ln_resvecs{j} = ln_resvec_q;
    Xmax_each(j) = max(Xu);
end
% Truncation
Lmin = floor(min(Xmax_each));
resvec_matrix = NaN(Lmin, m); % construct convergence history for all rhs
for j = 1:m
    resvec_matrix(:, j) = interp_ln_resvecs{j}(1:Lmin);
end
if anynan(resvec_matrix)
    fprintf("\n%s solover, precond=%g\n", solver, precond);
    error("NaN in conv history");
end
    
mean_log = mean(resvec_matrix, 2); %'omitnan'); 
geo_mean = exp(mean_log);
end
