function [params, adam_param] = adam(params, grads,beta1, beta2,t,adam_param, lr, e)


for i = 1:length(params)
        adam_param{i,1} = beta1.*adam_param{i,1} + (1-beta1).*grads{i};
        adam_param{i,2} = beta2.*adam_param{i,2} + (1-beta2).*(grads{i}).^2;
        v = adam_param{i,1}/(1-beta1^t);
        s = adam_param{i,2}/(1-beta2^t);
        params{i} = params{i}  - (lr * v)./(sqrt(s)+e);
end


end





