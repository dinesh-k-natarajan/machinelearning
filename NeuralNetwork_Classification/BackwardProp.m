function [grad_W, grad_b] = BackwardProp(y,X,Z,f_beta,h,W,b,Activation,LossFn)
%% BackwardProp(y,X,Z,f_beta,h,W,b)
% INPUT: 
%       y      : Output class labels (1 x m array)
%       X      : cell containing activations of each layer (also inputs)
%       Z      : cell containing inputs to each layer 
%       f_beta : array containing predicted output values
%       h      : array indicating the structure of the neural network using 
%                the size of each layer
%       W      : cell containing the initialised weight matrices
%       b      : cell containing the initialised biases
%   Activation : Type of activation function to be used
%       LossFn : Type of loss function to be used    
% OUTPUT:
%       grad_W : cell containing gradients w.r.t the weights of each layer
%       grad_b : cell containing gradient w.r.t the biases of each layer

%% Body
% Initialization
m      = size(y,2);
L      = size(h,2)-1;
grad_W = cell(L,1);
grad_b = cell(L,1);
del    = cell(L,1);

% Calculation of gradients w.r.t 'z' of all layers
if strcmp(LossFn,'Hinge')
    y(y==0)  = -1; % convert y to -1's and 1's
    del{L,1} = (-y.*(1-y.*f_beta'>0))'; % gradient of hinge loss
elseif strcmp(LossFn, 'NLL')
    del{L,1} = (sigmoid(f_beta')-y)'; % gradient of binary NLL loss 
else
    error('Error: Invalid Loss function, check assigned loss function type');
end

for i=L-1:-1:1
   del{i,1} = del{i+1,1} * W{i+1,1} .* DerivativeActivationFunction(Z{i,1},Activation)';
end

% Calculation of gradients w.r.t parameters of NN + Summing up
for i=1:L
   grad_W{i,1} = (del{i,1}'*X{i,1}')/m;  
   grad_b{i,1} = del{i,1}';  
   grad_b{i,1} = sum(grad_b{i,1},2)/m;
end

end

