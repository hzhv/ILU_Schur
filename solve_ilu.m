function y = solve_ilu(l,u,p,bs,x)
	%%% Return U\(L\x)

	colors = sort(unique(p));
	% Do [l00 0  ] \ [x0] = [l00\x0                 ]
	%    [l10 l11]   [x1]   [l11\(-l10*(l00\x0)+x1) ]
	y = zeros(size(x));
	for c=colors(:)'
		e = l(p==c,p==c)\x(p==c,:);
		y(p == c,:) = e;
		x(p > c,:) = x(p > c,:) - l(p>c,p==c)*e;
	end
	x = y;
	% Do [u00 u01] \ [x0] = [u00\(x0-u01*(u11\x1) ]
	%    [0   u11]   [x1]   [u11\x1               ]
	for c=colors(end:-1:1)'
		e = u(p==c,p==c)\x(p==c,:);
		y(p == c,:) = e;
		x(p < c,:) = x(p < c,:) - u(p<c,p==c)*e;
	end
end
