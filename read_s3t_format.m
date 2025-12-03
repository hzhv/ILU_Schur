function [A, nat_dim_rows, nat_dim_cols, metadata] = read_s3t_format(varargin)
	% Read coarse operators generated with chroma/mgproton
	%
	% A = READ_S3T_FORMAT('file0.s3t', 'filet1.set', ...) returns a sparse matrix
	% in natural ordering stored on all the input files.

	for mode=0:1
	for fi = 1:nargin
		% Open file
		f = fopen(varargin{fi}, 'r');

		% Read magic number, it should be 314
		magic_number = fread(f, 1, 'int32');
		if magic_number ~= 314
			error("The file is not in S3T format: invalid magic number");
		end

		% Read version number, it should be 0
		version = fread(f, 1, 'int32');
		if version ~= 0
			error("unsupported version of S3T format");
		end

		% Read value type: one of
		%- 0: float
		%- 1: double
		%- 2: complex float
		%- 3: complex double
		value_type = fread(f, 1, 'int32');
		if value_type < 0 || value_type > 3
			error("The file is not in S3T format: invalid value type");
		end
		if value_type == 0 || value_type == 2
			prec = 'single';
		else
			prec = 'double';
		end
		if value_type <= 1
			cmpl = 1;
		else
			cmpl = 2;
		end

		% Read checksum
		checksum_type = fread(f, 1, 'int32');
		if checksum_type < 0 || checksum_type > 2
			error("The file is not in S3T format: invalid checksum type");
		end

		% Read the number of dimensions
		ndim = fread(f, 1, 'int32');
		if ndim < 0
			error("The file is not in S3T format: invalid number of dimensions");
		end
		assert(ndim == (2+4+4+1)*2)
		
		% Read metadata size
		metadata_size = fread(f, 1, 'int32');
		if metadata_size < 0
			error("The file is not in S3T format: invalid metadata size");
		end

		% Read metadata content
		metadata = fread(f, metadata_size, 'char');
		metadata = char(metadata(:)');

		% Read padding
		fread(f, mod(8 - mod(metadata_size, 8), 8), 'char');

		% Read tensor size (from slowest index to fastest index)
		dim = fread(f, ndim, 'double');
		dim = dim(end:-1:1);

		% Read checksum block (ignored)
		fread(f, 1, 'double');

		% Read the number of chunks
		num_chunks = fread(f, 1, 'double');

		% Read the chunks and store them on the output matrix
		dim_rows = transpose(dim(1:ndim/2));
		dim_cols = transpose(dim(ndim/2+1:end));
		nat_dim_rows = tonat_size(dim_rows, dim_rows);
		nat_dim_cols = tonat_size(dim_cols, dim_cols);
		if mode == 1 && fi == 1
			%A = spalloc(prod(nat_dim_rows), prod(nat_dim_cols), nz);
			ai = zeros(nz, 1);
			aj = zeros(nz, 1);
			av = zeros(nz, 1);
			nz = 0;
		elseif mode == 0 && fi == 1
			nz = 0;
		end
		for chunk = 1:num_chunks
			% Read number of blocks
			num_blocks = fread(f, 1, 'double');

			% Read the ranges
			from_size = fread(f, ndim*2*num_blocks, 'double');
			from_size = reshape(from_size, ndim, 2, num_blocks);
			from_size = from_size(end:-1:1,:,:);
			rows_from_size = from_size(1:ndim/2,:,:);
			cols_from_size = from_size(ndim/2+1:end,:,:);

			for i=1:num_blocks
				% Read values
				values = fread(f, cmpl*prod(from_size(:,2,i)), 'double');
				if cmpl == 2
					values = complex(values(1:2:end), values(2:2:end));
				end
				nat_rows_from_i = tonat_coor(rows_from_size(:,1,i), dim_rows);
				nat_rows_size_i = tonat_size(rows_from_size(:,2,i), dim_rows);
				nat_cols_from_i = tonat_coor(cols_from_size(:,1,i), dim_cols);
				nat_cols_size_i = tonat_size(cols_from_size(:,2,i), dim_cols);
				row_idx = coor2index(index2coor(0:prod(nat_rows_size_i)-1,nat_rows_size_i)+nat_rows_from_i, nat_dim_rows);
				col_idx = coor2index(index2coor(0:prod(nat_cols_size_i)-1,nat_cols_size_i)+nat_cols_from_i, nat_dim_cols);
				if mode == 1
					nrows = numel(row_idx);
					ncols = numel(col_idx);
					ai(nz+(1:nrows*ncols), :) = repmat(row_idx+1, ncols, 1);
					aj(nz+(1:nrows*ncols), :) = kron(col_idx+1, ones(nrows,1));
					av(nz+(1:nrows*ncols), :) = values;
					%A(row_idx+1, col_idx+1) = reshape(values, numel(row_idx), numel(col_idx));
				end
				nz = nz + numel(row_idx)*numel(col_idx);
			end

			% Read chunk checksum (and ignore)
			if checksum_type == 2
				fread(f, num_blocks, 'double');
			end

			if mod(chunk-1,ceil(num_chunks/100)) == 0
				fprintf("%d%% ", round(chunk/num_chunks*100));
			end
		end
		fprintf("\n");

		% Read global checksum (and ignore)
		if checksum_type > 0
			fread(f, 1, 'double');
		end

		fclose(f);
	end
	end
	A = sparse(ai, aj, av, prod(nat_dim_rows), prod(nat_dim_cols));
end

function coor = index2coor(index, dim)
	dim = dim(:)';
	n = numel(dim);
	p = zeros(1,n);
	p(1) = 1;
	for i = 2:n
		p(i) = p(i-1)*dim(i-1);
	end
	coor = mod(floor(index(:)./p), dim);
end

function index = coor2index(coor, dim)
	dim = dim(:)';
	n = numel(dim);
	assert(size(coor,2) == n);
	p = zeros(n,1);
	p(1) = 1;
	for i = 2:n
		p(i) = p(i-1)*dim(i-1);
	end
	index = mod(coor + dim, dim) * p;
end

function nat_coor = tonat_coor(coor, dim)
	coor = coor(:)';
	dim = dim(:)';
	assert(all(coor < dim))
	assert(numel(coor) == 2+4+4+1)
	nat_coor = zeros(1,6);
	nat_coor(1:2) = coor(1:2);
	deblock = coor(3:6) + coor(7:10).*dim(3:6);
	deblock(1) = deblock(1)*dim(11) + mod(sum(deblock(2:4)) + coor(11), dim(11));
	nat_coor(3:6) = deblock;
end

function nat_coor = tonat_size(s, dim)
	s = s(:)';
	dim = dim(:)';
	assert(numel(s) == 2+4+4+1)
	assert(dim(1) == 1 || s(3:11) == ones(1,9) || s(3:11) == dim(3:11))
	nat_coor = zeros(1,6);
	nat_coor(1:2) = s(1:2);
	deblock = s(7:10).*dim(3:6);
	if s(11) == dim(11)
		deblock(1) = deblock(1)*dim(11);
	end
	nat_coor(3:6) = deblock;
end
