function [isOK, badColors, where] = check_eo_compatibility(Colors, D, N)
% Check the Oddity and Coloring Compatibilty
% Outputs:
%   isOK       : Bool 
%   badColors  : array, colors that against the compatibilty
%   where      : cell，idx of badColors (part of them)
    
    % Get eo reording permuation vector
    eo = zeros(N,1);
    coords = cell(1, numel(D));
    for i = 1:N
        [coords{:}] = ind2sub(D, i);
        x = cell2mat(coords) - 1;
        eo(i) = mod(sum(x), 2);
    end

    maxC = max(Colors);
    badColors = [];
    where = {};
    for c = 1:maxC
        idx = find(Colors == c);
        if isempty(idx), continue; end
        par = unique(eo(idx));
        if numel(par) > 1 % so not same Oddity
            badColors(end+1) = c;
            s = struct();
            s.even_idx = idx(eo(idx)==0);
            s.odd_idx  = idx(eo(idx)==1);

            s.even_idx = s.even_idx(1:min(5,end));
            s.odd_idx  = s.odd_idx(1:min(5,end));
            where{end+1} = s; 
        end
    end

    isOK = isempty(badColors);
end
