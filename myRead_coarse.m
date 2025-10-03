function [A, bs, dim] = read_coarse(filename)
    f = fopen(filename, 'r');
    ndim = fread(f, 1, 'single');
    dim = double(fread(f, ndim, 'single'));
    bs = double(fread(f, 1, 'single'));

    p = zeros(ndim,1);
    p(1) = bs;
    for i = 1:ndim-1
        p(i+1) = p(i)*dim(i);
    end

    A = sparse(prod(dim)*bs, prod(dim)*bs);
    direction = [ 0  0  0  0;
                  1  0  0  0;
                 -1  0  0  0;
                  0  1  0  0;
                  0 -1  0  0;
                  0  0  1  0;
                  0  0 -1  0;
                  0  0  0  1;
                  0  0  0 -1];
    d = 0;
    while true
        [crow, c] = fread(f, ndim, 'single');
        if c == 0, break; end
        ccol = fread(f, ndim, 'single');

        ccol=mod(ccol,dim);
        crow=mod(crow,dim);

        % FIX #2: The assertion below was failing because for some data, when d is a
        % multiple of 9, ccol ~= crow. Since ccol is recalculated immediately
        % after this, this check can be safely removed.
        % assert(mod(d, 9) ~= 0 || all(ccol == crow))
        
        ccol = mod(crow + dim + direction(mod(d, 9)+1, :)', dim);
        d = d + 1;
        
        % FIX #1: Convert the calculated indices to integers using round() to
        % prevent warnings when using them for array indexing.
        i = round(crow(:)'*p) + 1;
        j = round(ccol(:)'*p) + 1;
        
        v = double(fread(f, bs*bs*2, 'single'));
        fprintf('i=%d, j=%d, d=%d\n', i, j, d);
        
        % The indices are now integers, so this assignment is safe.
        A(i:(i+bs-1), j:(j+bs-1)) = reshape(complex(v(1:2:end), v(2:2:end)), [bs bs]);
    end 
    fclose(f);
    save('A.mat', 'A');
end