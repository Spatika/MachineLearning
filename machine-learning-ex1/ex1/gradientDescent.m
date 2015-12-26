function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


   del1 = 0 ;
   del2 = 0 ;
   h = 0 ;
   
    for i = 1:m
        for j = 1:length(theta)
            h = h + theta(j).*X(i,j) ;
        end   
        del1 = del1 + (h - y(i, 1)).*X(i, 1);
        del2 = del2 + (h - y(i, 1)).*X(i, 2) ;
        h = 0 ;
    end
    
    coeff = alpha./m ;
    theta(1) = theta(1) - coeff.*del1 ;
    theta(2) = theta(2) - coeff.*del2 ;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
