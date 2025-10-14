function coor = index2coor(index, dim)
	n = numel(dim);
	p = zeros(1,n);
	% p(end) = 1;
	% for i = n:-1:2
	% 	p(i-1) = p(i)*dim(i);
	% end
	p(1) = 1;
	for i = 2:n
		p(i) = p(i-1)*dim(i-1);
	end
	coor = mod(floor(index(:)./p), dim);
end
