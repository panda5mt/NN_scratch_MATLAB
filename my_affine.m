%% Affineレイヤ
classdef my_affine
    properties
        W
        b
        x
        
        dW
        db
        
    end
    methods
        function obj = my_affine(W, b)
            if nargin == 2
                
                obj.W = W;
                obj.b = b;
                obj.dW = [];
                
            end
        end
            
        function [obj, out] = forward(obj, x)            
%             size(x)
%             size(obj.W)
            obj.x = x;            
            out = obj.W * obj.x + obj.b;
            
        end
        
        function [obj, dx] = backward(obj, dout)
            %size(obj.x)
            dx = obj.W' * dout;
            obj.dW = dout * obj.x';
            obj.db = sum(dout, 2);
            
        end
    end
end


