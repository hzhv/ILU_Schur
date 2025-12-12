function p = partitioning(dims, bs, doms)
% partition the Lattice based on the dims and domains 
% 
% Return the domain index for each row
%   dims: Lattice Dimension               e.g. [4 4 4 8]
%   bs  : block size on the diagonal,     e.g. 64
%   doms: # of blocks in each dimenstion, e.g. [1 1 2 4], 2 blocks on 3rd
%         Dimension, 4 blocks on 4th dimension

    D = dims./doms; 
    coords = index2coor(0:prod(dims)-1, dims);

    domain_coords = floor(coords./D); 
    domain_index = coor2index(domain_coords, doms);

    p = kron(domain_index, ones(bs, 1));
end

