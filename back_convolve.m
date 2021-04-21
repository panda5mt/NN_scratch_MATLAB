% backprop through convolutional layer

function [C, dconv1, db] = back_convolve(conv1, dconv2,stride, padding,w)


data = conv1;
if sum(padding) > 0
p_data = padarray(data, padding,0);
else
    p_data = data;
end

%derivative with respect to the weight
C= zeros(size(w,1), size(w,2), size(w,3), size(w,4), size(p_data,4));
for b = 1:size(p_data, 4)
for j = 1:stride(1):size(p_data,1)-size(w,1)+1
for k = 1:stride(2):size(p_data,2)-size(w,2)+1
    for f = 1:size(w,4)
        for ch = 1:size(w,3)
            C(:,:,ch,f,b) = C(:,:,ch,f,b) + (conv1(j:j+size(w,1) -1, k:k+size(w,2)-1,ch,b)).*dconv2(j,k,f,b);
        end
    end
end

end
end

%derivative with respect to input
dconv1 = zeros(size(conv1));
for b = 1:size(conv1,4)
for i = 1:size(dconv2,1)
    for j = 1:size(dconv2,2)
        for f = 1:size(w,4)
            dconv1(i:i+size(w,1)-1, j:j+size(w,2)-1,:,b) = dconv1(i:i+size(w,1)-1, j:j+size(w,2)-1,:,b)  + ((w(:,:,:,f)*dconv2(i,j,f,b)));
        end
    end
end
end
db = dconv1;

end


