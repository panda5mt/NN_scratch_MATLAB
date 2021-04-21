
%for padding = 1; stride = 1
function res = vectorized_conv(x,kern, b)
    rc = floor(((size(x(:,:,1,1)) - size(kern(:,:,1,1)))) + 1);
    ind = reshape(1:size(x,1)*size(x,2),size(x,1),size(x,2));
    %C = im2col(ind, size(kern(:,:,1,1)), 'sliding');
    C = im2col_sliding(ind, size(kern(:,:,1,1)));

    n = numel(x(:,:,1,1));

    l = 0:size(kern,3)-1;
    l = l*n;
    l = reshape(l,1,1,length(l));

    C = repmat(C, 1,1, size(kern,3));
    C = C + l;
    
    C = repmat(C,1,1,1,size(x,4));
    l2 = 0:size(x,4)-1;
    l2 = l2*n*size(x,3);
    %l2 = repmat(C(end,end,end,1),1,1,1,size(x,4));
    l2 = reshape(l2,1,1,1,length(l2));
    C = C+l2;
    
    xC = x(C);

    k = reshape(kern,numel(kern(:,:,1,1)),1,size(kern,3),size(kern,4));
    k = repmat(k,1,1,1,1,size(x,4));
    xC = reshape(xC, size(xC,1),size(xC,2),size(xC,3),1,size(xC,4));
    res = sum(sum(xC.*k,1),3);
    res = reshape(res, rc(1), rc(2),size(kern,4), size(x,4));
    res = res + b;
end

