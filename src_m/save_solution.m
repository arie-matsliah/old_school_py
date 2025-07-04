%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function save_solution(filename,P)

[i,j] = find(P);
fid = fopen(filename,'w');
fprintf(fid,'Male Node ID,Female Node ID\n'); 
fprintf(fid,'m%d,f%d\n',[i j]');              
fclose(fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
