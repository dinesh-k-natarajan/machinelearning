function [W_opt,b_opt,X_opt,Z_opt,f_beta_opt,Loss_opt] = GradDesc(data,h,W,b,f_beta,grad_W,grad_b,alpha,lambda,Activation)
%% GradDesc(data,h,W,b,grad_W,grad_b,alpha)
% INPUT:
%       data        : Inputs and Output labels from data set
%       h           : array indicating the structure of the neural network using 
%                     the size of each layer
%       W           : cell containing the initialised weight matrices
%       b           : cell containing the initialised biases
%       f_beta      : array containing predicted output values
%       grad_W      : cell containing gradients w.r.t the weights of each layer
%       grad_b      : cell containing gradient w.r.t the biases of each layer
%       alpha       : learning rate for the gradient descent algorithm
%       lambda      : regularization parameter
%       Activation  : Type of activation function to be used
% OUTPUT:
%       W_opt       : cell containing optimized weight matrices
%       b_opt       : cell containing optimized bias arrays
%       X_opt       : cell containing optimized activations of each layer (incl. inputs)
%       Z_opt       : cell containing optimized inputs to each layer 
%       f_beta_opt  : array containing predicted output values
%       Loss_opt    : optimum loss obtained after gradient descent
%
%% Body
% Initialization
L = size(h,2)-1;
tol   = 1e-4;
max_iter = 2.5e5;
count = 1;
delta = ones(L,2);
figure('visible','on');clf;movegui('center');hold on;
title('Convergence of Gradient Descent');
xlabel('Number of iterations');ylabel('Hinge Loss');
loss = HingeLoss(f_beta,data(:,3),W,lambda);

% Update of NN parameters
while (count<max_iter && mean(delta(:))>tol)
    for i=1:L
        W{i,1}     = W{i,1} - alpha * grad_W{i,1} - (lambda/size(data,1))*W{i,1};
        delta(i,1) = norm(grad_W{i,1})/numel(grad_W{i,1});
        b{i,1}     = b{i,1} - alpha * grad_b{i,1};
        delta(i,2) = norm(grad_b{i,1})/numel(grad_b{i,1});
    end
    [X,Z,f_beta]    = ForwardProp(data(:,1:2)',h,W,b,Activation);
    [grad_W,grad_b] = BackwardProp(data(:,3)',X,Z,f_beta,h,W,b,Activation);
    if ~mod(count,100)
        loss0 = loss;
        loss  = HingeLoss(f_beta,data(:,3),W,lambda); 
        plot([count-100 count],[loss0 loss], 'k-');
        fprintf('Iteration %d, Norm = %0.5f, Loss = %0.4f\n',count,mean(delta(:)),loss);
    end
    count = count + 1;
end

% Extracting values from the terminated while loop
loss = HingeLoss(f_beta,data(:,3),W,lambda); 
plot(count, loss, 'r.','MarkerSize',20);
fprintf('Iteration %d, Norm = %0.5f, Loss = %0.4f\n',count,mean(delta(:)),loss);
W_opt      = W;
b_opt      = b;
X_opt      = X;
Z_opt      = Z;
f_beta_opt = f_beta;
Loss_opt   = loss;

end

