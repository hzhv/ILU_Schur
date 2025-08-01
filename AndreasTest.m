clc; clear; 
a = lap_kD_periodic([8 8 8 8],1); 
n=size(a,1);
a = a + speye(n)*1e-7;
fprintf(' Cond Number of A %g\n',cond(full(a)));
[l,u]=ilu(a); kond = cond(full(u\(l\a)));
figure; spy(full(l*u));
fprintf(' Cond Number of inv(LU)*A, original order (n colors) = %g\n',kond);
eo = 1:n;
for k=1:4
     % colors = greedyColor(a(eo,eo)^k); % Prof. A's
     [colors, nColor] = displacement_even_odd_coloring_nD_lattice([8 8 8 8], k, [0 0 0 0]);
     % order = color2perm(colors);
     [~, order] = sort(colors);

     figure; spy(a(eo(order),eo(order)))

     [l,u]=ilu(a(eo(order),eo(order)));
     kond = cond(full(u\(l\a(eo(order),eo(order)))));
     fprintf('Cond of inv(LU)*A(p,p) with %d dist, %d colors =%g\n',k,max(colors)+1,kond);
     if k==1, eo = order; end
end

