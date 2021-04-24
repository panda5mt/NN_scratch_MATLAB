close all;
clc;
%choose batch size and number of epochs for training
batch_size = 50;
epochs = 1000;

%choose parameters for adam
beta1 = 0.95;
beta2 = 0.99;
lr = 0.0001; %learning rate
eps = 1e-8;
t = 0;

%% Load MNIST
%load 'mnist.mat';
%% Load MNIST END

%% Load CIFAR-10
load './cifar-10-batches-mat/data_batch_1.mat'
XTrain = uint8(zeros(size(data,1), 32, 32, 3)); % N H W C
YTrain = labels;
for i=1:size(data,1)
   data_in = data(i,:);
   dataColor = reshape(data_in,[32, 32, 3]);
   XTrain(i,:,:,:) = dataColor;
end
XTrain = permute(XTrain, [2 3 4 1]); % H W C N

%% Load CIFAR-10 END

params = init_param(); % initialize weights

p_adam = cell(length(params),2); %initialize adam algorithm
for i = 1:length(params)
    p_adam{i,1} = zeros(size(params{i}));
    p_adam{i,2} = zeros(size(params{i}));
end

X = 0;
Y = 0;

prob_total = 0;
batch_total = 0;

for j = 1:epochs
    %% epoch数が進行するにつれて学習率を減らしていく
    lr = lr / (1.01^(j))
    
    if lr < 9e-9 
        lr = 9e-9;
    end
    sh_i = randperm(size(XTrain,4)); %shuffle training data
    XTrain = double(XTrain(:,:,:,sh_i));
    YTrain = double(YTrain(sh_i));

for i = 1:batch_size:size(XTrain,4)
    im = XTrain(:,:,:,i:i+batch_size-1);
    lab = zeros(10,size(im,4));
    
    it = 0;
    for k = i:i+batch_size-1
     it = it +1;
     lab(double(YTrain(k))+1,it) = 1;
    end
    
t = t+1;

[L, grads,params, probs] = net_grad(params, im, lab); %calculate net output and gradients
[params, p_adam] = adam(params, grads,beta1, beta2,t,p_adam, lr, eps); %train using adam


prob_total = prob_total + probs;
batch_total = batch_total + batch_size;

if mod((i-1)/batch_size,20) == 0
    num = (i - 1)/batch_size;
    
    % 正解率計算
    pc_mean = prob_total / batch_total * 100;
    
    % 一旦リセット
    prob_total = 0;
    batch_total = 0;
    
    disp(['training accuracy: ',num2str(pc_mean),'%.']);
    disp([num2str(num) ' iterations. Loss: ' num2str(L)]);
    
    
    index = int16(num / 100 + 1);
    X_val = num + int16(size(XTrain,4)/batch_size * (j-1));
    X = [X X_val];
    Y = [Y pc_mean];
    
    %hold on;
    figure(1);
    plot (X,Y);
    title('CIFAR-10 Training Accuracy');
    xlabel('iterations');
    ylabel('accuracy[%]');
    drawnow
    %hold off;
end
end
end

