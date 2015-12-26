function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% y is output - cost in 10k
% X is matrix of instances - each row is an instance
h = 0 ;
sum = 0 ;

for i = 1:m
    for j = 1:length(theta)
        h = h + theta(j).*X(i,j);
    end ;   
    sum = sum + ((h-y(i, 1)).^2) ;
    h = 0 ; 
end

J = (0.5./m).*sum ; 



% =========================================================================

end
