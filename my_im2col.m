function col = my_im2col(input_data, filter_h, filter_w, stride, pad)
    %{
        input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
        filter_h : フィルターの高さ
        filter_w : フィルターの幅
        stride : ストライド
        pad : パディング

        Returns
        -------
        col : 2次元配列
    %}

    %% input_dataから、バッチサイズN, チャンネル数C, 画像高さH, 画像幅W を取得
    %input_data:[H, W, C, N]
    input_data = permute(input_data, [4 3 1 2]);
    [N, C, H, W] = size(input_data);
    
    %% 畳み込み処理後の出力の高さ out_h, 幅 out_w を計算
    out_h = fix((H + 2*pad - filter_h)/stride(1)) + 1;
    out_w = fix((W + 2*pad - filter_w)/stride(2)) + 1;
    
    % パディング
    img = padarray(input_data,[pad pad],0,'both');
    col = zeros(N, C, filter_h, filter_w, out_h, out_w);
    
    
    % 縦方向のスライシング範囲の最大値y_maxを求める
    for y=1:filter_h
        y_max = int16((y-1) + stride(2) * out_h);
        for x=1:filter_w

            % 横方向のスライシング範囲の最大値x_maxを求める
            x_max = int16((x-1) + stride(1) * out_w);

            % スライシング結果をゼロ行列に格納
            % y から y_max まで stride_h 間隔でスライシング
            % x から x_max まで stride_w 間隔でスライシング
            col(:, :, y, x, :, :) = img(:,:, y:stride(2):y_max, x:stride(1):x_max);
        end
    end
    col = permute(col,[1 5 6 2 3 4]);
    
    col = reshape(col, N * out_h * out_w,[]);
    
end