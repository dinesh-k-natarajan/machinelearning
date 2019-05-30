function [sigmoid] = sigmoid(z)
%% sigmoid(z)
% Computes the sigmoid function of variable z
% INPUT: 
%       z       - constant, vector or matrix
% OUTPUT:
%       sigmoid - sigmoid of z is defined as 1/1+e^-z 
%               - It has the same dimensions as z

%% Body
sigmoid = zeros(size(z));
sigmoid = 1./(1+exp(-z));
end