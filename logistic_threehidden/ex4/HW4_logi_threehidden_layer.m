clear ; close all; clc
input_layer_size  = 16;
hidden_layer_size1 = 8;
hidden_layer_size2 = 4;
hidden_layer_size3 = 8;
num_labels = 16; 
X = zeros(16,16);  
for i = 1:16
  X(i,i)=1
  end
y = 0:15;
y = y';
%nn_params = initial_nn_params;

%%%%%test begin%%%%%
%%%%%i = 1;
%%%%test end%%%%%

cd "C:/Users/Dai/Dropbox/02 classes/data mining/homework/HW4/logistic_threehidden/ex4"

t1=mktime (localtime (time ()));

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size1);
initial_Theta2 = randInitializeWeights(hidden_layer_size1, hidden_layer_size2);
initial_Theta3 = randInitializeWeights(hidden_layer_size2, hidden_layer_size3);
initial_Theta4 = randInitializeWeights(hidden_layer_size3, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:); initial_Theta4(:)];

save initial_Theta1_848_try5.mat initial_Theta1
save initial_Theta2_848_try5.mat initial_Theta2
save initial_Theta3_848_try5.mat initial_Theta3
save initial_Theta4_848_try5.mat initial_Theta4

fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 5000000);

%  You should also try different values of lambda
lambda = 0;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size1, ...
                                   hidden_layer_size2, ...
                                   hidden_layer_size3, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost, i] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size1 * (input_layer_size + 1)), ...
                 hidden_layer_size1, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size1 * (input_layer_size + 1))): (hidden_layer_size1 * (input_layer_size + 1)+hidden_layer_size2 *(hidden_layer_size1 + 1) )), ...
                 hidden_layer_size2, (hidden_layer_size1 + 1));

Theta3 = reshape(nn_params((1 + hidden_layer_size1 * (input_layer_size + 1)+ hidden_layer_size2 * (hidden_layer_size1 + 1)): (hidden_layer_size1 * (input_layer_size + 1)+hidden_layer_size2 * (hidden_layer_size1 + 1)+hidden_layer_size3 *(hidden_layer_size2 + 1) )), ...
                 hidden_layer_size3, (hidden_layer_size2 + 1));
                 
Theta4 = reshape(nn_params((1 + hidden_layer_size1 * (input_layer_size + 1)+ hidden_layer_size2 * (hidden_layer_size1 + 1)+ hidden_layer_size3 * (hidden_layer_size2 + 1)): end), ...
                 num_labels, (hidden_layer_size3 + 1));
                 
pred = predict(Theta1, Theta2, Theta3, Theta4, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double((pred-1) == y)) * 100);


X = [ones(16,1) X];
h1 = X';
zh2 = Theta1 * h1;
h2 = sigmoid(zh2);

h2 = [ones(1,16); h2];
h3 = sigmoid(Theta2 * h2);

h3 = [ones(1,16); h3];
h4 = sigmoid(Theta3 * h3);

h4 = [ones(1,16); h4];
h5 = sigmoid(Theta4 * h4);

t2 = mktime (localtime (time ()));
t = t2-t1;

save -text try5 i t h2 h3 h4 h5





