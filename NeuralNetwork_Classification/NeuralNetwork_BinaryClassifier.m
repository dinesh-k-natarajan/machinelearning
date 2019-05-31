%{

Filename   : NeuralNetwork_BinaryClassifier.m
Author     : Dinesh K. Natarajan
Created on : 20:08:16, 26 May 2019

%}

clear;
close all;
%% Pre-processing 

% Load data from text file
data=load('data2Class.txt');
% Randomly permute the data
data = data(randperm(size(data,1)),:);

%% Setup
% Neural Network Structure
h = [2 20 1]; % size of [input layer, hidden layer(s), output layer]
lambda = 0;   % regularization parameter (increase if NN is overfitting)
Activation = 'LeakyReLU'; % Activation function types: 'LeakyReLU', 'ReLU', 'Sigmoid', 'tanh'
LossFn = 'NLL'; % Loss function types: 'Hinge', 'NLL'
 
%% Initialization of Weights
[W,b] = InitializeWeights(h,'W_rand','b_rand');

% Forward Propagation
[X,Z,f_beta] = ForwardProp(data(:,1:2)',h,W,b,Activation); 

% Backward Propagation
[grad_W,grad_b] = BackwardProp(data(:,3)',X,Z,f_beta,h,W,b,Activation,LossFn);

% Gradient Descent Optimization of NN parameters
alpha = 0.01; % Learning rate for GD
[W_opt,b_opt,X_opt,Z_opt,f_beta_opt,Loss_opt] = GradDesc(data,h,W,b,f_beta,grad_W,grad_b,alpha,lambda,Activation,LossFn);

%% Accuracy Measure
pred = f_beta_opt;
pred(pred>0)=1;
pred(pred<0)=0;
target = data(:,3);
correctly_pred = pred==target;
accuracy = sum(correctly_pred(:))/size(data,1)*100;
fprintf('\nAccuracy Score: %0.2f%%\n',accuracy);

%% Visualizing the output of NN after one FP pass
figure;clf;hold on;
movegui('northwest');
scatter3(data(:,1),data(:,2),data(:,3),'r.');
view(3);
[a_plot,b_plot] = meshgrid(-2.5:.1:2.5,-2.5:.1:2.5);
[X_plot,Z_plot,f_beta_plot] = ForwardProp([a_plot(:),b_plot(:)]',h,W,b,Activation);
Xgrid = f_beta_plot;
Ygrid = sigmoid(Xgrid);
Ygrid = reshape(Ygrid,size(a_plot));
h1 = surface(a_plot,b_plot,Ygrid);
view(3);
grid on;
title('Prediction by the NN after one FP pass - \sigma(f_\beta(x)');

%% Visualizing the output of NN after FP and BP
figure;clf;hold on;
movegui('northeast')
scatter3(data(:,1),data(:,2),data(:,3),'r.');
[X_plot2,Z_plot2,f_beta_plot2] = ForwardProp([a_plot(:),b_plot(:)]',h,W_opt,b_opt,Activation);
Xgrid2 = f_beta_plot2;
Ygrid2 = sigmoid(Xgrid2);
Ygrid2 = reshape(Ygrid2,size(a_plot));
h2 = surface(a_plot,b_plot,Ygrid2);
view(3);
grid on;
title('Prediction by the NN after FP and BP - \sigma(f_\beta(x)');

%% Using the NN for prediction

test_input = rand(h(1),1); % (d x m) array, d: input dim, m: data set size
fprintf('\nTest_input values: %0.4f',test_input);
[~,~,test_pred] = ForwardProp(test_input,h,W_opt,b_opt,Activation);
test_pred_label = cell2mat({test_pred>0}).*1; % pred label as 1 if pred >0, 0 if pred < 0. 
fprintf('\nPredicted value for test_input: %d\n', test_pred_label);
