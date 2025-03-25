function G = sigmoid_nn(U,V)
% G = sigmoid_nn(U,V) is a global function handle created to pass 
% parameters gamma and alpha to the custom SVM kernel Sigmoid Function.

global khandle
G = khandle(U,V);

