%% 交差エントロピー誤差による損失関数
function y = cross_entropy_error(inp, label)
    
% もし正解ラベルがone-hot表記でない場合は直す
    if(numel(inp) ~= numel(label))
        label = (label==1:size(inp,1))';
    end
    
    y = -sum(label.*log(max(inp, 1e-8)));     
    y = mean(y); %mean over batches
end