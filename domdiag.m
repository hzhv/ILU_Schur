function r=domdiag(a,p)
	n = size(a,1);
	assert(n == numel(p))
	r = sparse([], [], [], n, n, nnz(a));
	d = unique(p);
	for i=d(:)'
		r(p==i,p==i) = a(p==i,p==i);
	end
end
