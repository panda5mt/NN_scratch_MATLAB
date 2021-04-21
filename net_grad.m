%% 各レイヤの順伝播と逆伝播の計算

function [L, grads, params, probs] = net_grad(params, image, label)

%% パラメータ準備
W1 = params{1};
W2 = params{2};
W3 = params{3};
W4 = params{4};

b1 = params{5};
b2 = params{6};
b3 = params{7};
b4 = params{8};

%% レイヤの設定
Conv1       = my_Convolution(W1, b1, [1 1], 0);
Relu1       = my_relu();
Conv2       = my_Convolution(W2, b2, [1 1], 0);
Relu2       = my_relu();
Affine1     = my_affine(W3, b3);        
Relu3       = my_relu();
Affine2     = my_affine(W4, b4);
lastLayer   = my_SoftmaxWithLoss();

%% 順伝播
[Conv1, conv1]      = Conv1.forward(image);
[Relu1, a1]         = Relu1.forward(conv1);
[Conv2, conv2]      = Conv2.forward(a1);
[Relu2, conv2]      = Relu2.forward(conv2);

m1 = maxpool(conv2, [2,2], [1 1], 0);
fc = reshape(m1, size(m1,1)*size(m1,2)*size(m1,3),size(m1,4));

[Affine1,out]       = Affine1.forward(fc);
[Relu3,out]         = Relu3.forward(out);
[Affine2,out]       = Affine2.forward(out);
[lastLayer, ~]      = lastLayer.forward(out, label);

%% 認識結果
probs = lastLayer.y;

% ラベル取得(MNISTの場合1~10のいずれか) 
[~, probs] = max(probs,[],1);
% 正解ラベル取得(MNISTの場合1~10のいずれか) 
[~, lab] = max(label,[],1);

%% 正解率
probs = sum(probs == lab,'all');
%% 損失
L                   = lastLayer.loss;

%% 逆伝播
dout = 1;
[lastLayer, dout]   = lastLayer.backward(dout);
[Affine2, dout]     = Affine2.backward(dout);
[Relu3, dout]       = Relu3.backward(dout);
[Affine1, dfc]      = Affine1.backward(dout);

dpool = reshape(dfc, size(m1));
dout = maxpool_backprop(conv2, dpool, [2 2] , [1 1], 0);
db2 = sum(sum(sum(dout,1),2),4);

[Relu2, dout]       = Relu2.backward(dout);
[Conv2, dout]       = Conv2.backward(dout);
[Relu1, dout]       = Relu1.backward(dout);
[Conv1, ~]          = Conv1.backward(dout);

dW4 = Affine2.dW;
db4 = Affine2.db;
dW3 = Affine1.dW;
db3 = Affine1.db;
dW2 = Conv2.dW;
db1 = Conv2.db;
dW1 = Conv1.dW;

grads = {dW1, dW2, dW3, dW4, db1, db2, db3, db4};
params = {W1, W2, W3, W4, b1, b2, b3, b4};
end

