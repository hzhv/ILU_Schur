function r=invblkdiag(a,bs)
	n = size(a,1);
	assert(mod(n,bs) == 0)
	r = sparse([], [], [], n, n, bs*n);
	for i=1:n/bs
		i0 = (i-1)*bs+(1:bs);
		r(i0,i0) = inv(a(i0,i0));
	end
end
