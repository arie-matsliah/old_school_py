%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function P = read_solution(filename)

% SLURP FILE
fid = fopen(filename,'r');
data = textscan(fid,'%c%d%c%c%d','HeaderLines',1);
fclose(fid);

% CREATE PERMUTATION MATRIX
i = data{2}(:); % male node
j = data{5}(:); % female node
P = sparse(i,j,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
