function p = coloring(dim, bs, power, step, disp)

	% Compute the neighbors
	nei = zeros(1, numel(dim));
	for i = 1:power
		nei = unique([nei; neighbors(nei, step, dim)], 'rows');
	end
	nei = unique([nei+disp; nei-disp], 'rows');

	% Use greedy coloring doing first even and then odd
	n = prod(dim);
	p = -ones(n,1);
	first_color = 0;
	for oddity=0:1
		for i=1:n
			coor=index2coor(i-1,dim);
			if mod(sum(coor),2) ~= oddity, continue; end
			colors = unique(p(coor2index(coor+nei,dim)+1));
			c = first_color;
			while nnz(c == colors) > 0
				c = c + 1;
			end
			p(i) = c;
		end
		first_color = max(p);
	end

	p = kron(p, ones(bs,1));
end

function b = neighbors(coor, step, dim)
	assert(size(coor,2) == numel(dim))
	n = numel(dim);
	k = nnz(dim == 2) + nnz(dim > 2)*2; % total # of nei
	ncoor = size(coor,1);
	b = zeros(k*ncoor, n);
	j = 1;
	for i=1:n
        keyboard
		if dim(i) >= 2*step
			b(j:k:end,:) = coor;
			b(j:k:end,i) = mod(b(j:k:end,i)+step, dim(i));
			j = j + 1;
		end
		if dim(i) > 2*step
			b(j:k:end,:) = coor;
			b(j:k:end,i) = mod(b(j:k:end,i)-step+dim(i), dim(i));
			j = j + 1;
		end
	end
end

