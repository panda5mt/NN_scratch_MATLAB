function params = init_param()

W1 = 0.01 * randn(5, 5, 3, 8); %FH,FW,C,FN
b1 = zeros(1,8);
W2 = 0.01 * randn(5, 5, 8, 3);
b2 = zeros(1,3);
W3 = 0.01 * randn(3, 3, 3, 16);
b3 = zeros(1,16);
W4 = 0.01 * randn(128,7056);
b4 = zeros(128, 1);
W5 = 0.01 * randn(10,128);
b5 = zeros(10,1);

params = {W1,W2,W3,W4,W5,b1,b2,b3,b4,b5};