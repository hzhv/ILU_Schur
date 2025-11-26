function [is_diag_dom, ratio] = check_diag_dominance(A)
    diag_vals = abs(diag(A));
    
    row_sums = sum(abs(A), 2);
    
    off_diag_sums = row_sums - diag_vals;
    
    ratio = diag_vals ./ (off_diag_sums + eps); 
    
    strict_rows = diag_vals > off_diag_sums;
    weak_rows   = diag_vals >= off_diag_sums;

    is_diag_dom = all(strict_rows);
    fprintf('  #Rows strictly dominant: %g, %g%%\n', nnz(strict_rows), nnz(strict_rows)/size(A,1));

    
    % figure; 
    % plot(1:size(A,1), diag_vals, 'b.', 'DisplayName', '|a_{ii}|'); hold on;
    % plot(1:size(A,1), off_diag_sums, 'r.', 'DisplayName', '\Sigma_{j\ne i}|a_{ij}|');
    % legend; title('Diagonal Dominance Check');
    % xlabel('Row Index');
end