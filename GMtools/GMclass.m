% gm = GMclass(X,Y,cov_type) Train Gaussian classifier
%
% Given a labelled training set (X,Y), this computes the maximum likelihood
% estimator for a Gaussian classifier, namely:
% - The prior probabilities (proportion of each class).
% - The mean vectors (average of each class).
% - The covariance matrices (covariance of each class, suitably averaged when
%   the covariance is shared across classes).
%   Note: we ensure each covariance matrix is full-rank by adding to its
%   diagonal C*eps where C is set below. You might want to adjust this value
%   in practice.
% The result is returned in the GMtools format for a Gaussian mixture, where
% each component corresponds to one class.
% See GMpdf.m for descriptions of some of the arguments below.
%
% In:
%   X: NxD matrix containing N D-dim points rowwise.
%   Y: Nx1 matrix containing the class labels for X (in 1..M).
%   cov_type: covariance type (one of 'F','f','D','d','I','i').
% Out:
%   gm: Gaussian mixture struct.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.
%
% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function gm = GMclass(X,Y,cov_type)

[N,D] = size(X);
allM = unique(Y)'; M = allM(end);		% classes

% Initialise parameters
gm.type = cov_type; gm.p = zeros(M,1); gm.c = NaN(M,D);
switch cov_type
 case 'i', gm.S = 0;
 case 'I', gm.S = NaN(M,1);
 case 'd', gm.S = zeros(1,D);
 case 'D', gm.S = NaN(M,D);
 case 'f', gm.S = zeros(D,D);
 case 'F', gm.S = NaN(D,D,M);
end

for m = allM
  idx = find(Y==m); L = length(idx);		% data points in class m
  if L > 0
    gm.p(m) = L;				% prior probability
    gm.c(m,:) = mean(X(idx,:),1)';		% mean vector
    switch cov_type				% covariance matrix
     case 'i', gm.S = gm.S + L*mean(var(X(idx,:),1));
     case 'I', gm.S(m) = mean(var(X(idx,:),1));
     case 'd', gm.S = gm.S + L*var(X(idx,:),1);
     case 'D', gm.S(m,:) = var(X(idx,:),1);
     case 'f', gm.S = gm.S + L*cov(X(idx,:),1);
     case 'F', gm.S(:,:,m) = cov(X(idx,:),1);
    end
  end
end

% Normalise counters
gm.p = gm.p/N;
switch cov_type
 case {'i','d','f'}, gm.S = gm.S/N;
end

% Ensure each covariance matrix is full-rank:
C = 1e4;
switch cov_type				% covariance matrix
 case {'i','I','d','D'}, gm.S(gm.S==0) = C*eps;
 case 'f', gm.S = gm.S + C*eps*speye(D,D);
 case 'F', for m=1:M, gm.S(:,:,m) = gm.S(:,:,m) + C*eps*speye(D,D); end
end

