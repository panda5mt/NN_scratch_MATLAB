function img = my_col2im(col, input_shape, filter_h, filter_w, stride, pad)
    %{
        input_shape : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
        filter_h : フィルターの高さ
        filter_w : フィルターの幅
        stride : ストライド
        pad : パディング

        Returns
        -------
        col : 2次元配列
    %}

    %% input_shapeから、バッチサイズN, チャンネル数C, 画像高さH, 画像幅W を取得
    %input_shape:[H, W, C, N]
    input_shape = permute(input_shape, [4 3 1 2]);
    [N, C, H, W] = size(input_shape);

    %% 畳み込み処理後の出力の高さ out_h, 幅 out_w を計算
    out_h = fix((H + 2*pad - filter_h)/stride) + 1;
    out_w = fix((W + 2*pad - filter_w)/stride) + 1;
    
    col = reshape(col, N, out_h, out_w, C, filter_h, filter_w);
    col = permute(col,[1 4 5 6 2 3]);
    
    % パディング
    img = zeros(N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1);
    
    
    % 縦方向のスライシング範囲の最大値y_maxを求める
    for y=1:filter_h
        y_max = int16((y-1) + stride * out_h);
        for x=1:filter_w

            % 横方向のスライシング範囲の最大値x_maxを求める
            x_max = int16((x-1) + stride * out_w);

            img(:, :, y:stride:y_max, x:stride:x_max) = col(:,:, y, x, :, :);
        end
    end
    
    if(pad ~= 0)
        img = img(:, :, pad:H+pad, pad:W+pad);
    else 
        img = img(:,:,H,W);
    end
end