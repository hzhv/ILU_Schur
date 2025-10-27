function [l,u] = ilu0_colors(a, p, bs)
%% implictly ilu0(A-colored)
	% [a00 a01] = [l00 0  ] [u00 u01] = [l00*u00  l00*u01        ]
    % [a10 a11]   [l10 l11] [0   u11]   [l10*u00  l10*u01+l11*u11]
	% [l00,u00] = lu(a00)
    % u01 = l00\a01
	% l10 = a10/u00
	% l11*u11 = a11 - l10*u01 = a11 - (a10/u00)*(l00\a01)

	if nargin<=2, bs=1; end
	n = size(a,1);
	assert(numel(p) == n)

	[ii,jj,~] = find(a);
	o = zeros(numel(ii),1);
	f = @(x) floor((x-1)/bs);
	iiu = f(ii) <= f(jj);
	u = sparse(ii(iiu),jj(iiu), o(iiu), n,n);
	iil = f(ii) > f(jj);
	l = sparse(ii(iil),jj(iil), o(iil), n,n, numel(iil)+n);

	colors = sort(unique(p));
	for c=colors(:)'
		n0 = nnz(p == c);
		l(p == c, p == c) = speye(n0);        % l_00, l_11
		u(p == c, p == c) = blkdiag(a(p == c, p == c), bs); %u_00, u_11
		u(p == c, p > c) = a(p == c, p > c);  % u_01
		l(p > c, p == c) = a(p > c, p == c) * invblkdiag(u(p == c, p == c), bs); % l_10 

		ij1 = p(ii) > c & p(jj) > c;
		ii1 = ii(ij1);
		jj1 = jj(ij1);
		b1 = unique(floor((ii1-1)/bs) + floor((jj1-1)/bs)*(n/bs));
		for i=b1(:)'
			i0 = mod(i,n/bs)*bs+(1:bs);
			j0 = floor(i/(n/bs))*bs+(1:bs);
			a(i0,j0) = a(i0,j0) - l(i0, p == c) * u(p == c, j0);
		end
	end
end
