function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

m = length(y); 
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    J_history(iter) = computeCost(X, y, theta);
    temp_theta = theta;
    for j = 1:length(theta)
        temp_theta(j) = theta(j) - alpha * computeCostDerivative(X, y, theta, j);
    end
    theta = temp_theta;
end

end

function result =  computeCostDerivative(X, y, theta, j)
    m = length(y);
    dcosts = (X*theta - y).*X(:, j);
    result =  sum(dcosts) / m;
end