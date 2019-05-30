function [X,Z,f_beta] = ForwardProp(x,h,W,b,Activation)
%% ForwardProp(x,W,b)
% INPUT: 
%       x      : 'd x m' transposed array of input data from data set
%       h      : array indicating the structure of the neural network using 
%                the size of each layer
%       W      : cell containing the initialised weight matrices
%       b      : cell containing the initialised biases
%  Activation  : Type of activation function to be used
% OUTPUT: 
%       X      : cell containing activations of each layer (also inputs)
%       Z      : cell containing inputs to each layer 
%       f_beta : array containing predicted output values
%
%% Body
% Initialization
L      = size(h,2);
X      = cell(L,1);
X{1,1} = x;
Z      = cell(L-1,1);

% Forward Propagation
for i=2:L
        Z{i-1,1} = W{i-1,1} * X{i-1,1} + b{i-1,1};
        X{i,1}   = ActivationFunction(Z{i-1,1},Activation);
end

f_beta = Z{end,1}';

end

