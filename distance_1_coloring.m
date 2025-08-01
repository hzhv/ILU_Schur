function [color, nColor] = distance_1_coloring(A)
n = size(A,1);
color = zeros(n,1);
nColor = 0; % total # of colors
for v = 1:n
    nbr = find(A(v,:)); % nnz neighbours of v
    used = color(nbr);
    c = 1;
    while any(used == c)
        c = c + 1;
    end
    color(v) = c;
    if c > nColor, nColor = c; end 

end
end % of coloring
