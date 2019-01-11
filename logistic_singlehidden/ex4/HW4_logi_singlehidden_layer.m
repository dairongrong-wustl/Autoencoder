clear ; close all; clc
input_layer_size  = 16;
hidden_layer_size = 3;
num_labels = 16; 
X = zeros(16,16);  
for i = 1:16
  X(i,i)=1
  end
y = 0:15;
y = y';

cd "C:/Users/Dai/Dropbox/02 classes/data mining/homework/HW4/logistic_singlehidden/ex4"

t1=mktime (localtime (time ()));
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

save initial_Theta1_3_try5.mat initial_Theta1
save initial_Theta2_3_try5.mat initial_Theta2

fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 200000);

%  You should also try different values of lambda
lambda = 0;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost, i] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double((pred-1) == y)) * 100);

X = [ones(16,1) X];
a1 = X';
z2 = Theta1 * a1;
a2 = sigmoid(z2);

a3 = [ones(1,16); a2];
a4 = sigmoid(Theta2 * a3);
t2 = mktime (localtime (time ()));
t = t2-t1;

save -text hidden_3_try5 i t a2 a4





