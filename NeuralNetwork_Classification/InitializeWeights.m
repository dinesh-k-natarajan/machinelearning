function [W,b] = InitializeWeights(h,W_dist,b_dist)
%% InitializeWeights(h)
% 
% INPUT:
%       h      : array indicating the structure of the neural network using 
%               the size of each layer
%       W_dist : string indicating the type of distribution for the weights
%       b_dist : string indicating the type of distribution for the biases
% 
% OUTPUT:
%       W      : cell containing all the weight matrices of the NN
%       b      : cell containing all the bias vectors of the NN
% 
%% Body
% Initialization
L = size(h,2);
W = cell(L-1,1);
b = cell(L-1,1);
%% Weights
if strcmp(W_dist,'W_rand')
   for i=2:L
       W{i-1,1} = normrnd(0,1/sqrt(h(i-1)),h(i),h(i-1));   
   end
elseif strcmp(W_dist,'W_zeros')
   for i=2:L
       W{i-1,1} = zeros(h(i),h(i-1));
   end
elseif strcmp(W_dist,'W_ones')
    for i=2:L
       W{i-1,1} = ones(h(i),h(i-1));
   end
else 
    error('Error in W_dist : use either W_rand, W_zeros, or W_ones');
end

%% Biases
if strcmp(b_dist,'b_rand')
   for i=2:L
       b{i-1,1} = -1 + 2*rand(h(i),1);
   end
elseif strcmp(b_dist,'b_zeros')
   for i=2:L
       b{i-1,1} = zeros(h(i),1);
   end
elseif strcmp(b_dist,'b_ones')
   for i=2:L
       b{i-1,1} = ones(h(i),1);
   end
else 
    error('Error in b_dist : use either b_rand, b_zeros, or b_ones');
end

end

