%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function C = read_connectome(filename)

% SLURP FILE
fid = fopen(filename,'r');
data = textscan(fid,'%c%d%c%c%d%c%f','HeaderLines',1);
fclose(fid);

% CREATE WEIGHT MATRIX
i = data{2}(:);   % from node
j = data{5}(:);   %   to node
w = data{7}(:);   %    weight
C = sparse(i,j,w);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
