close all;
clc;
%choose batch size and number of epochs for training
batch_size = 32;
epochs = 1;

%choose parameters for adam
beta1 = 0.95;
beta2 = 0.99;
lr = 0.001; %learning rate
eps = 1e-8;
t = 0;
load 'mnist.mat';
params = init_param(); % initialize weights

p_adam = cell(length(params),2); %initialize adam algorithm
for i = 1:length(params)
    p_adam{i,1} = zeros(size(params{i}));
    p_adam{i,2} = zeros(size(params{i}));
end

for j = 1:epochs
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

if mod((i-1)/batch_size,100) == 0
    num = (i - 1)/batch_size;
    pc_mean = probs / batch_size * 100;
    disp(['training accuracy: ',num2str(pc_mean),'%.']);
    disp([num2str(num) ' iterations. Loss: ' num2str(L)]);
end
end
end




