% Copyright (c) 2016 by Miguel A. Carreira-Perpinan
% for use in CSE176 Introduction to Machine Learning at UC Merced

% Dataset for 1D regression: a curve f(x) with additive Gaussian noise
% with zero mean and variance s^2.

rng(1778);					% Fix random number gen. seed
N = 100;					% Number of points
D = 1;						% Dimension of the inputs x
DD = 1;						% Dimension of the outputs y
a = 4; b = -0.5; f = @(x) sin(a*x) + b;		% Nonlinear function
s = 0.1;					% Noise standard deviation
% Training set
X = 3*rand(N,D);				% Input points x
F = f(X);					% Exact outputs f(x)
Y = F + s*randn(size(F));			% Noisy outputs y

% This is for plotting the true function f
XX = linspace(-1,4,1000)'; FF = f(XX);
ax = [-1 4 -2 1];	% Range for plot

% Plot
figure(1); plot(XX,FF,'r-',X,Y,'bo','MarkerSize',12);
axis(ax); daspect([1 1 1]); box on; xlabel('x'); ylabel('y');


% Suggestions of things to try:
% - Change the parameters of the function f or the type of function itself.
% - Change the level of noise.

