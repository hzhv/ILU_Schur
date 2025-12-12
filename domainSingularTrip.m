function domainSingularTrip(A, dims, bs, dom, k, tol, maxit)
    disp('Partitioning domain...');
    p = partitioning(dims, bs, dom);
    A = domdiag(A, p);

    unique_doms = unique(p);
    num_doms = length(unique_doms);
    
    k_sub = ceil(k/num_doms);

    u_rows_cell = cell(num_doms, 1);
    u_cols_cell = cell(num_doms, 1);
    u_vals_cell = cell(num_doms, 1);
    
    v_rows_cell = cell(num_doms, 1);
    v_cols_cell = cell(num_doms, 1);
    v_vals_cell = cell(num_doms, 1);
    
    s_vals_all = []; 
    
    global_col_offset = 0; % Column Shift for each domain's sing.vector

    disp('Starting local SVDs on each domain...');
    for i = 1:num_doms
        dom_id = unique_doms(i);
        
        idx = find(p == dom_id); % global idx
        A_sub = A(idx, idx);
        assert(length(idx) == size(A_sub, 1), ...
            "S_domain_size=%d larger than A_domain_size=%d", k_sub, size(A_sub,1));
        [u, s, v] = getSingularTrip(A_sub, k_sub, tol, maxit);
        
        s = diag(s);

        [r_local, c_local, val_local] = find(u);
        u_rows_cell{i} = idx(r_local);
        u_cols_cell{i} = global_col_offset + c_local;
        u_vals_cell{i} = val_local;

        [r_local, c_local, val_local] = find(v);
        v_rows_cell{i} = idx(r_local);
        v_cols_cell{i} = global_col_offset + c_local;
        v_vals_cell{i} = val_local;
        
        s_vals_all = [s_vals_all; s];
        
        global_col_offset = global_col_offset + k_sub;
    end
    disp('Local SVDs done. Stitching results...');

    % Concat all domains
    u_rows = vertcat(u_rows_cell{:}); % [a; b]
    u_cols = vertcat(u_cols_cell{:});
    u_vals = vertcat(u_vals_cell{:});
    
    v_rows = vertcat(v_rows_cell{:});
    v_cols = vertcat(v_cols_cell{:});
    v_vals = vertcat(v_vals_cell{:});
    

    U = sparse(u_rows, u_cols, u_vals, size(A,1), global_col_offset);
    V = sparse(v_rows, v_cols, v_vals, size(A,2), global_col_offset);

    SCell = {U, s_vals_all, V};
    save("singularTripL2_DD_Approx.mat", "SCell");
    disp("Done.");
end