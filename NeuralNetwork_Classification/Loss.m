function [loss] = Loss(f,y,W,lambda,LossFn)
%% Loss(f,y)
% INPUT:
%       f    : Output values predicted by the NN (m x 1 vector)
%       y    : Output class labels from data set (m x 1 vector)
%       W    : Cell containing all the weights
%      lambda: Regularization parameter
%     LossFn : Type of loss function to be used    
% OUTPUT:
%       loss : Hinge Loss function defined as Loss = max(0,1-f*y)
%              or Binary Cross Entropy or neg-log-likelihood loss function
%
%% Body
% Assemble all weights into one array
weights = [];
for i=1:size(W,1)
    temp = W{i,1}(:);
    weights = [weights;temp];
end

% loss
if strcmp(LossFn,'Hinge')
    % convert y into a vector of -1's and 1's
    y(y==0)  = -1;
    loss = sum(max(0,1-y.*f))/size(y,1)+(lambda/(2*size(y,1)))*(norm(weights));
elseif strcmp(LossFn,'NLL')
    loss = (-sum(y.*log(sigmoid(f))+(1-y).*log(1-sigmoid(f)))/size(y,1))...
            +(lambda/(2*size(y,1)))*(norm(weights));
else
    error('Error: Invalid Loss function, check assigned loss function type');
end
end

