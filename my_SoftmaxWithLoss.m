%% SoftmaxWithLossレイヤ
classdef my_SoftmaxWithLoss
    properties
        loss = []
        y = []
        t = []
       
    end
    methods
%         function obj = my_SoftmaxWithLoss()
%             if nargin == 0
%                 
%                 obj.t = [];
%                 obj.y = [];
%                 obj.loss = [];
%                 
%             end
%         end
        function [obj,loss] = forward(obj, x, t)
            
            obj.t = t;
            obj.y = my_softmax(x);
            
            obj.loss = cross_entropy_error(obj.y, obj.t);
            loss = obj.loss;
            
            
        end
        
        function [obj, dx] = backward(obj, dout)
            if dout ~= 1
                dout = 1; 
            end
            batch_size = size(obj.t,1);
            
            if numel(obj.t) == numel(obj.y) %% one-hot?
                dx = (obj.y - obj.t) / batch_size;
            else
                dx = (obj.y - obj.t);
            end
        end
    end
end



