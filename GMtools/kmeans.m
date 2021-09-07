% [C,L,e,code] = kmeans(X,K[,init,maxit,tol]) K-means clustering
%
% In:
%   X: NxD matrix containing N D-dimensional data points rowwise.
%   K: integer in [1,N] containing the desired number of clusters.
%   init: initialisation, one of the following (default: 'kmeans++'):
%      'kmeans++': Arthur & Vassilvitskii SODA 2007;
%      'rndlabels': L = random assignment;
%      'rndmeans': C = random means in the range of X;
%      'rndsubset': C = random subset of data points;
%      C: KxD matrix containing the initial K cluster means.
%      Note: kmeans(X,K,'kmeans++',0) provides the kmeans++ initialisation.
%   maxit: maximal number of iterations. Default: Inf.
%   tol: small positive number, tolerance in the relative change of the
%      error function to stop iterating. Default: 0.
% Out:
%   C: KxD matrix containing the K cluster means.
%   L: Nx1 list containing the cluster labels (1 to K). Thus,
%      L(n) = k says point Xn is in cluster k.
%   e: list of values (for each iteration) of the error function.
%      This is the sum over all clusters of the within-cluster sums of
%      point-to-mean distances.
%   code: stopping code: 0 (tolerance achieved), 1 (maxit exceeded).
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2017 by Miguel A. Carreira-Perpinan

function [C,L,e,code] = kmeans(X,K,init,maxit,tol)

% ---------- Argument defaults ----------
if ~exist('init','var') || isempty(init) init = 'kmeans++'; end;
if ~exist('maxit','var') || isempty(maxit) maxit = Inf; end;
if ~exist('tol','var') || isempty(tol) tol = 0; end;
% ---------- End of "argument defaults" ----------

% Initialisation
C = km_init(X,K,init); [L,e] = km_le(X,C);
if maxit == 0, code = 1; return; end

% Iteration
code = -1; i = 0;
while code < 0
  oldL = L; oldC = C;			% Keep if back up needed
  C = km_m(X,L,C);			% New means,...
  [L,e(end+1)] = km_le(X,C);		% ...labels and error value
  i = i + 1;
  % Stopping condition
  if e(end) >= (1-tol)*e(end-1), code = 0;
  elseif i >= maxit, code = 1;
  end
end

% Back up if the last iteration increased the error function. This can
% happen due to numerical inaccuracy if two points Xn are very close.
if e(end) > e(end-1), i = i - 1; e = e(1:end-1); C = oldC; L = oldL; end


% [L,e] = km_le(X,C) Labels and error function given means
%
% Input/output arguments as above.
% Out:
%   e: value of the error function.

function [L,e] = km_le(X,C)

[e,L] = min(sqdist(X,C),[],2); e = sum(e);


% C = km_m(X,L,C[,deadC]) Means given labels
%
% Input/output arguments as above.
% In:
%   deadC: what to do with a dead mean (which has no associated data points),
%      one of 'rnd' (assign to it a random data point) or 'leave' (leave as
%      is). Default: 'rnd'.
% Out:
%   C: new means.

function C = km_m(X,L,C,deadC)

% Argument defaults
if ~exist('deadC','var') || isempty(deadC) deadC = 'rnd'; end;

for k=1:size(C,1)
  tmp = mean(X(L==k,:),1);
  if ~isnan(tmp(1))
    C(k,:) = tmp;
  elseif strcmp(deadC,'rnd')
    % Dead mean: assign to it one data point at random
    C(k,:) = X(randi(size(X,1)),:);
    % Otherwise:
    % Leave dead mean as is (works badly because often means end up dead)
  end
end


% C = km_init(X,K,init) Initialise means
%
% Input/output arguments as above.

function C = km_init(X,K,init)

if ischar(init)
  [N,D] = size(X);
  switch init
   case 'kmeans++'	% C = subset of data points encouraging dispersion
    C = zeros(K,D); D2 = Inf(1,N); C(1,:) = X(randi(N),:);
    for i=2:K
      % update shortest distance for each point given new centroid
      dist = sqdist(C(i-1,:),X); D2(dist<D2) = dist(dist<D2);
      % sample new centroid propto D2
      cD2 = cumsum(D2); C(i,:) = X(sum(cD2(end)*rand>=cD2)+1,:);
    end
   case 'rndlabels'	% labels = random assignment
    C = km_m(X,[(1:K)';randi(K,N-K,1)],zeros(K,D));
   case 'rndmeans'	% C = uniform random points in the range of X
    m = min(X,[],1); M = max(X,[],1);
    C = bsxfun(@plus,m,bsxfun(@times,M-m,rand(K,D)));
   case 'rndsubset'	% C = random subset of data points
    % Works badly because often means compete inside clusters
    C = X(randperm(N,K),:);
  end
else
  C = init;		% C = user-provided
end

