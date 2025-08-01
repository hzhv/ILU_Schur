clear; clc;

%% 1. 构造示例稀疏下三角矩阵 A (n×n)
n = 8;                       % 矩阵维度
rng(0);                     % 固定随机种子，保证可重复
A = sprand(n,n,0.3);        % 生成稀疏随机矩阵 (密度 30%)
A = tril(A);                % 保证矩阵为下三角，便于前向替换演示
% 调整对角元素，使矩阵严格对角占优（增强稳定性）
for i = 1:n
    A(i,i) = sum(abs(A(i,:))) + 1;
end
spy(A)
%% 2. 指定块大小 bs 并划分 Block
bs = 2;                     % 块大小：每 bs 行形成一个块
numBlocks = ceil(n/bs);     % 计算块数（最后一块可小于 bs）
blocks = cell(numBlocks,1);
for p = 1:numBlocks
    rows = ( (p-1)*bs + 1 ) : min(p*bs, n);
    blocks{p} = rows;       % blocks{p} 保存第 p 个块所包含的行索引
end

%% 3. 构造 Block 依赖图 (超点图)
adj = sparse(numBlocks, numBlocks);  % 邻接矩阵初始化
for p = 1:numBlocks
    for q = 1:numBlocks
        if p == q, continue; end
        rows = blocks{p}; cols = blocks{q};
        % 提取块 p 的行, 块 q 的列子矩阵
        subA = A(rows, cols);
        [i_idx, j_idx, ~] = find(subA);
        % 全局行/列索引
        global_i = rows(i_idx);
        global_j = cols(j_idx);
        % 如果存在 i > j，即行 i 依赖于列 j，则块间有依赖
        if any(global_i > global_j)
            adj(p,q) = 1;
        end
    end
end

%% 4. 对块依赖图进行 Greedy 多色着色
colors = zeros(numBlocks,1);
for p = 1:numBlocks
    used = colors(logical(adj(p,:))); % 邻居块已占用的颜色
    c = 1;
    while ismember(c, used)
        c = c + 1;
    end
    colors(p) = c;             % 为块 p 分配最小可用颜色
end
numColors = max(colors);

%% 5. 输出块划分及对应颜色
fprintf('=== BMC 块划分与颜色 ===\n');
fprintf('块大小 bs = %d, 块数 = %d, 颜色数 = %d\n', bs, numBlocks, numColors);
for p = 1:numBlocks
    fprintf('  Block %d: rows %s  →  color %d\n', p, mat2str(blocks{p}), colors(p));
end

%% 6. BMC 前向替换示例 (模拟 Gauss-Seidel)
b = rand(n,1);              % 随机右端向量 b
y = zeros(n,1);             % 初始化解向量 y

% 按颜色顺序迭代：同色块可并行
for c = 1:numColors
    idxBlocks = find(colors == c);
    % 对于每个同色块，块内按行依次做前向替换
    for p = idxBlocks'
        for ii = blocks{p}
            % y(ii) = (b(ii) - sum_{j<ii}(A(ii,j)*y(j))) / A(ii,ii)
            y(ii) = (b(ii) - A(ii,1:ii-1) * y(1:ii-1)) / A(ii,ii);
        end
    end
end

fprintf('\n完成 BMC 前向替换，示例解向量 y = \\n');
disp(y);
