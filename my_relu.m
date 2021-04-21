%% Rectified Linear Unit関数による活性化関数
classdef my_relu
    properties
        mask % 行列中の要素で0以下を[1],0より大きいなら[0]にする
        
    end
    methods
%         function obj = my_relu(m)
%             if nargin == 1
%                 
%                 obj.Value = z;
%             end
%         end
        
        function [obj, out] = forward(obj, x)
            obj.mask = (x <= 0);
            %out = x .* ~(obj.mask);
            out = max(0,real(x)) + 1i*(max(0, imag(x)));
            
        end
        
        function [obj, dx] = backward(obj, dout)
            dx = (~obj.mask) .* dout; 
        end
    end
end
