function C = input_convolve2d(x, W, b, stride, pad)

if sum(pad) > 0
    p_data = padarray(x, pad,0);
else
    p_data = x;
end

C = zeros([(size(x(:,:,1,1)) - size(W(:,:,1,1)) + 2*pad)./stride + 1, size(W,4), size(x,4)]); %initialize output array
ind1 = stride(1):size(p_data,1)-size(W,1)+1;
ind2 = stride(2):size(p_data,2)-size(W,2)+1;


for batch = 1:size(x,4) %batch size
for j = 1:length(ind1)
for k = 1:length(ind2)
for channel = 1:size(W,4) % number of filters

    C(j,k,channel,batch) = sum(p_data(ind1(j):ind1(j)+size(W,1) -1, ind2(k):ind2(k)+size(W,1) -1,:,batch).*W(:,:,:,channel), 'all');

end
end
end
end
C = C + b;


