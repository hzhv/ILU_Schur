function s = getSchurComplement(A, A_dim, A_bs)
    %  Get Schur Complement:
    %      s = @(x) a00*x - a01*(inva11*(a10*x))
    p = coloring(A_dim, A_bs, 1, 1, zeros(size(A_dim)));

    a00 = A(p==0,p==0);
    a01 = A(p==0,p==1);
    a10 = A(p==1,p==0);
    a11 = A(p==1,p==1);

    assert(nnz(blkdiag(a00, A_bs)-a00) == 0) 
    inva11 = invblkdiag(a11,A_bs);
    s = a00 - a01*(inva11*(a10));
end