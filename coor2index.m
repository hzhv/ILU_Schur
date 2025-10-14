function index = coor2index(coor, dim)
	dim = dim(:)';
	n = numel(dim);
	assert(size(coor,2) == n);
	p = zeros(n,1);
	% p(end) = 1;
	% for i = n:-1:2
	% 	p(i-1) = p(i)*dim(i);
	% end
	p(1) = 1;
	for i = 2:n
		p(i) = p(i-1)*dim(i-1);
	end
	index = mod(coor + dim, dim) * p;
end
