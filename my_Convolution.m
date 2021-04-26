%% Convolutionレイヤ
%  pad = 1; obj.obj.stride = 1 に固定
classdef my_Convolution
    properties
        W
        b
        stride
        pad
        x
        out
        dW
        db
    end
    methods
        function obj = my_Convolution(W, b, stride, pad)
            if nargin == 4
                
                obj.W = W;
                obj.b = b;
                obj.stride = stride;
                obj.pad = pad;
                
            end
        end
            
        function [obj, out] = forward(obj, x)
            % W (FH,FW,C,FN)
            
            % パディング
            if sum(obj.pad) > 0
                p_data = padarray(x, obj.pad, 0);
            else
                p_data = x;
            end

            C = zeros([(size(x(:,:,1,1)) - size(obj.W(:,:,1,1)) + 2*obj.pad)./obj.stride + 1, size(obj.W,4), size(x,4)]); %initialize output array
            ind1 = obj.stride(1):size(p_data,1)-size(obj.W,1)+1;
            ind2 = obj.stride(2):size(p_data,2)-size(obj.W,2)+1;


            for batch = 1:size(x,4) %batch size
            for j = 1:length(ind1)
            for k = 1:length(ind2)
            for channel = 1:size(obj.W,4) % number of filters

                C(j,k,channel,batch) = sum(p_data(ind1(j):ind1(j)+size(obj.W,1) -1, ...
                    ind2(k):ind2(k)+size(obj.W,1) -1,:,batch).*obj.W(:,:,:,channel), ...
                        'all');

            end
            end
            end
            end
%             size(C)
%             size(obj.b)
            C = C + obj.b;
            out = C;
            obj.out = C;
        end

        function [obj, dx] = backward(obj, dout)

            obj.dW = dw_back(obj.out, dout);

            kern = rot90(obj.W,2);
            dout = padarray(dout, [size(kern,1)-1, size(kern,2)-1],0);
            dx = vectorized_conv(dout,permute(kern, [1 2 4 3]), 0);
            obj.db = sum(sum(dx,1),2);
            obj.db = sum(obj.db(end));
            obj.dW = sum(obj.dW(end));

        end
    end
end



