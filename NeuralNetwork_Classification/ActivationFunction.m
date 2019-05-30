function [sigma] = ActivationFunction(z, type)
%% ActivationFunction(z)
% Computes the sigmoid function of variable z
% INPUT: 
%       z       - constant, vector or matrix
%       type    - string indicating type of activation function
% OUTPUT:
%       sigma   - Acitvation Function of z 
%               - It has the same dimensions as z

%% Body
sigma = zeros(size(z));
if strcmp(type,'LeakyReLU')
    sigma = max(0.01.*z,z);
elseif strcmp(type,'ReLU')
    sigma = max(0,z);
elseif strcmp(type,'Sigmoid')
    sigma = 1./(1+exp(-z));
elseif strcmp(type,'tanh')
    sigma = tanh(z);
else
    error('Error: Invalid Activation function, check defined activation type');
end
end