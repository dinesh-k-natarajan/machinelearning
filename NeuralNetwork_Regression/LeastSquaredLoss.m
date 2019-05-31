function [loss] = LeastSquaredLoss(f,y,W,lambda)
%% LeastSquaredLoss(f,y,W,lambda)
% INPUT:
%       f    : Output values predicted by the NN (m x 1 vector)
%       y    : Output values from data set (m x 1 vector)
%       W    : Cell containing all the weights
%      lambda: Regularization parameter
% OUTPUT:
%       loss : Least Squared Loss function defined as Loss = norm(f-y) +
%              regularization term
%
%% Body
% Assemble all weights into one array
weights = [];
for i=1:size(W,1)
    temp = W{i,1}(:);
    weights = [weights;temp];
end

% loss
loss = norm(f-y)/size(y,1)+(lambda/(2*size(y,1)))*(norm(weights));
end

