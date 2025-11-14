function getSS
% get Singular TripLet
A = load("A_level2.mat").A;
bs = 64; dim=[4 4 4 8];
p = coloring(dim,bs,1,1,zeros(size(dim)));


a00 = A(p==0,p==0);
a01 = A(p==0,p==1);
a10 = A(p==1,p==0);
a11 = A(p==1,p==1);

assert(nnz(blkdiag(a00, bs)-a00) == 0) 
inva11 = invblkdiag(a11,bs);
s = a00 - a01*(inva11*(a10));

disp("Start...")
[USch, SSch, VSch] = getSingularTrip(s,64,0.01,1000);
end