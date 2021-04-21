function out = im2col_distinct(A,blocksize)

nrows = blocksize(1);
ncols = blocksize(2);
nele = nrows*ncols;

row_ext = mod(size(A,1),nrows);
col_ext = mod(size(A,2),ncols);

padrowlen = (row_ext~=0)*(nrows - row_ext);
padcollen = (col_ext~=0)*(ncols - col_ext);

A1 = zeros(size(A,1)+padrowlen,size(A,2)+padcollen);
A1(1:size(A,1),1:size(A,2)) = A;

t1 = reshape(A1,nrows,size(A1,1)/nrows,[]);
t2 = reshape(permute(t1,[1 3 2]),size(t1,1)*size(t1,3),[]);
t3 =  permute(reshape(t2,nele,size(t2,1)/nele,[]),[1 3 2]);
out = reshape(t3,nele,[]);

return;