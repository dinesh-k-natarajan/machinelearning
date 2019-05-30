function [sigma_prime] = DerivativeActivationFunction(z,type)
%% DerivativeActivationFunction(z)
% Computes the sigmoid function of variable z
% INPUT: 
%       z             - constant, vector or matrix
%       type          - type of Activation function
% OUTPUT:
%       sigma_prime   - Derivative of activation function of z
%                     - It has the same dimensions as z

%% Body
sigma = zeros(size(z));
sigma_prime = zeros(size(z));
if strcmp(type,'LeakyReLU')
    sigma = max(0.01.*z,z);
    sigma_prime(sigma>0) = 1;
    sigma_prime(sigma<0) = 0.01;
elseif strcmp(type,'ReLU')
    sigma = max(0,z);
    sigma_prime(sigma>0)= 1;
elseif strcmp(type,'Sigmoid')
    sigma = 1./(1+exp(-z));
    sigma_prime = sigma.*(1-sigma);
elseif strcmp(type,'tanh')
    sigma_prime = 1 - tanh(z).^2;
else
    error('Error: Invalid Activation function, check defined activation type');
end
end