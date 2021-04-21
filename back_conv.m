% backprop through convolutional layer

function [dw, dx, db] = back_conv(conv1, dout, W)
%[dw2, dconv1, db1] = back_conv(conv1, dconv2, w2)

% data = conv1;
% if sum(padding) > 0
% p_data = padarray(data, padding,0);
% else
%     p_data = data;
% end

%derivative with respect to the weight

%input_convolve2d(data, kern, bias,stride, padding)


% dw = zeros([size(w), size(conv1,4)]);
% 
% 
% ind1 = 1:size(conv1,1)-size(dconv2,1)+1;
% ind2 = 1:size(conv1,2)-size(dconv2,2)+1;
% 
% for b = 1:size(dconv2,4) %batch size
% for f = 1:size(dconv2,3) % number of filters
% for j = 1:length(ind1)
% for k = 1:length(ind2)
% for ch = 1:size(conv1,3)
%     [j,k,ch,f,b]
%   dw(j,k,ch,f,b) = sum(conv1(ind1(j):ind1(j)+size(dconv2,1) -1, ind2(k):ind2(k)+size(dconv2,2) -1,ch,b).*dconv2(:,:,f,b), 'all');
% 
% 
% end
% end
% end
% end
% end

dw = dw_back(conv1,dout);


kern = rot90(W,2);
dout = padarray(dout, [size(kern,1)-1, size(kern,2)-1],0);

%dconv1 = dconv_back(dconv2,kern);
dx = vectorized_conv(dout,permute(kern, [1 2 4 3]), 0);
% ind1 = 1:size(dconv2,1)-size(kern,1)+1;
% ind2 = 1:size(dconv2,2)-size(kern,2)+1;
% dconv1 = zeros(size(conv1));
% 
% 
% for b = 1:size(dconv2,4) %batch size
% for j = 1:length(ind1)
% for k = 1:length(ind2)
% for ch = 1:size(kern,3)
% 
%   dconv1(j,k,ch,b) = sum(dconv2(ind1(j):ind1(j)+size(kern,1) -1, ind2(k):ind2(k)+size(kern,2) -1,:,b).*squeeze(kern(:,:,ch,:)), 'all');
% 
% 
% end
% end
% end
% end



db = sum(sum(dx,1),2);

end
