function [A, bs, dim] = read_coarse(filename)
	f = fopen(filename, 'r');
	ndim = fread(f, 1, 'single');
	dim = double(fread(f, ndim, 'single'));
	bs = double(fread(f, 1, 'single'));

	p = zeros(ndim,1);
	p(1) = bs;
	% for i = ndim:-1:2
	% 	p(i-1) = p(i)*dim(i);
	% end
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
		% assert(all(ccol < dim) && all(crow < dim) && all(ccol >= 0) && all(crow >= 0));
        %assert(all(ccol >= 0) && all(crow >= 0));
		ccol=mod(ccol,dim);
        crow=mod(crow,dim);
        assert(mod(d, 9) ~= 0 || all(ccol == crow))
		ccol = mod(crow + dim + direction(mod(d, 9)+1, :)', dim);
		d = d + 1;
		i = crow(:)'*p;
		j = ccol(:)'*p;
		v = double(fread(f, bs*bs*2, 'single'));
		fprintf('i=%d, j=%d, d=%d\n', i, j, d);
		A((1:bs)+i, (1:bs)+j) = reshape(complex(v(1:2:end), v(2:2:end)), [bs bs]);
	end	
	fclose(f);
end
