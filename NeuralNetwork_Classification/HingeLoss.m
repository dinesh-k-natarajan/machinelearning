function [loss] = HingeLoss(f,y,W,lambda)
%% HingeLoss(f,y)
% INPUT:
%       f    : Output values predicted by the NN (m x 1 vector)
%       y    : Output class labels from data set (m x 1 vector)
%       W    : Cell containing all the weights
%      lambda: Regularization parameter
% OUTPUT:
%       loss : Hinge Loss function defined as Loss = max(0,1-f*y)
%
%% Body
% Assemble all weights into one array
weights = [];
for i=1:size(W,1)
    temp = W{i,1}(:);
    weights = [weights;temp];
end
% convert y into a vector of -1's and 1's
y(y==0)  = -1;
% loss
loss = sum(max(0,1-y.*f))/size(y,1)+(lambda/(2*size(y,1)))*(norm(weights));
end

