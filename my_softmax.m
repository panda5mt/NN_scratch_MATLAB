%% softmax関数による活性化関数
function soft_out = my_softmax(inp)
    % オーバフロー対策
    inp = inp - max(inp);
    
    soft_out = exp(inp);
    soft_out = soft_out ./ sum(soft_out);
end