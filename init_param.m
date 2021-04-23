function params = init_param()

W1 = 1/sqrt(5*5*8)*randn(5, 5, 3, 8); %FH,FW,C,FN
b1 = zeros(1,1,8);
W2 = 1/sqrt(5*5*8)*randn(5,5, 8, 3);
b2 = 0;
W3 = 1/sqrt(128*1587)*randn(128,1587);
b3 = zeros(128, 1);
W4 = 1/sqrt(10*128)*randn(10,128);
b4 = zeros(10,1);

params = {W1,W2,W3,W4,b1,b2,b3,b4};