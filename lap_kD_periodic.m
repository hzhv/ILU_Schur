%
% Two cases: 
%
%   *   A = lap_kD_periodic(pts)
% 
%       Generates a sparse adjacency matrix (including the diagonal) of the 
%       graph of a dims-dimensional uniform, torus grid with of dimensions:
%	pts(1), pts(2), ..., pts(dims)
% 	Ex: A = lap_kD_periodic([4 4 4 8]);
%
%   *   A = lap_kD_periodic(pts,pde)
% 
%       If pde argument is present with any value, the function generates 
%	a sparse Laplacian on the pts dims-dimensional grid, with periodic 
%	boundary conditions. The stencil is (-1...-1 2*dims -1...-1).
%	Therefore the matrix is singular.
%	pts(1), pts(2), ..., pts(dims)
% 	Ex: A = lap_kD_periodic([4 4 4 8],1);  
%
function A = lap_kD_periodic(pts,pde)
dims = length(pts);
plane = cumprod(pts);
x = zeros(dims,1);
N = prod(pts);
A = spalloc(N,N,N*dims*2); % (dims*2 + 1) ?

% Natural order
for i=1:N 
   A(i,i) = 1;
   
   % Find the coordinates of node i first 
   x = index2coord(i,pts);

   for d=1:dims
        % 2 neighbors at every dim
	    x1=x;x2=x; 
	    x1(d)=mod(x(d)+1,pts(d)); % mod is for periodic
	    x2(d)=mod(x(d)-1,pts(d));
	    i1 = coord2index(x1,pts);
	    i2 = coord2index(x2,pts);
	    A(i,i1) = 1; A(i,i2) = 1; 
   end
end

% If we need a PDE Laplacian:
if (nargin == 2)
   A=-A; % Make all entries -1.
   A = spdiags(2*dims*ones(N,1),0,A); % replaces main diagonal of A with 2*dims
end

end % of lap_kD_periodic
%%
function i = coord2index(x,pts)
   dims = length(pts);
   i=x(1);
   c=1;
   for d=2:dims
      c=c*pts(d-1);
      i=i+c*x(d);
   end
   i=i+1;  % back to 1,2, ordering
end

function x = index2coord(i,pts)
   dims = length(pts);
   c = i-1; % consider 0,1,2.... ordering
   for d=1:dims-1
      x(d) = mod(c,pts(d));
      c=floor((c-x(d))/pts(d));
   end
   x(dims) = c;  %still 0,1, ordering
end