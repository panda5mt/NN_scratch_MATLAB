function params = init_param()

W1 = 1/sqrt(3*3*2)*randn(3, 3, 3, 8); %FH,FW,C,FN
b1 = zeros(1,1,8);
W2 = 1/sqrt(3*3*2)*randn(3, 3, 8, 3);
b2 = 0;
W3 = 1/sqrt(3*3*1)*randn(3, 3, 1, 16);
b3 = 0;
W4 = 1/sqrt(128*10000)*randn(128,10000);
b4 = zeros(128, 1);
W5 = 1/sqrt(10*128)*randn(10,128);
b5 = zeros(10,1);

params = {W1,W2,W3,W4,W5,b1,b2,b3,b4,b5};