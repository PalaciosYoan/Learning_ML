% [gm,e] = GMEM(X,M,cov_type[,gm,tol,maxit]) Gaussian mixture EM training
%
% Trains a Gaussian mixture model by expectation-maximisation (EM).
% See GMpdf.m for descriptions of some of the arguments below.
%
% The singularity problem is avoided by resetting small covariances to a
% constant value (not a very good solution, but generally works).
% Part of this code has been adapted from the Netlab toolbox.
%
% In:
%   X: NxD matrix containing N D-dim points rowwise. 
%   M: number of Gaussian mixture components.
%   cov_type: covariance type (one of 'F','D','d','I','i').
%   gm: initial parameters. Default: obtain from k-means.
%   tol: minimum relative increase in log-likelihood to keep iterating.
%      Default: 1e-5.
%   maxit: maximum number of iterations. Default: 1000.
% Out:
%   gm: Gaussian mixture struct.
%   e: list of negative log-likelihood values.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.
%
% Copyright (c) 2011 by Miguel A. Carreira-Perpinan and Chao Qin

function [gm,e] = GMEM(X,M,cov_type,gm,tol,maxit)

% ---------- Argument defaults ----------
if ~exist('gm','var') || isempty(gm) gm = GMinit(X,M,cov_type); end;
if ~exist('tol','var') || isempty(tol) tol = 1e-5; end;
if ~exist('maxit','var') || isempty(maxit) maxit = 1000; end;
% ---------- End of "argument defaults" ----------

mu = gm.c; S = gm.S; pm = gm.p; cov_type = gm.type;	% Extract GM fields
[N,D] = size(X);

MIN_COV = eps;			% Minimum singular value of covariance matrix
init_cov = S;			% Initial covariance matrix given by k-means

% Main loop of algorithm
nll_old = Inf; e = [];
for n = 1:maxit
    
  gm.c = mu; gm.S = S; gm.p = pm; gm.type = cov_type;
  
  % E-step
  
  [p,~,pm_x] = GMpdf(X,gm,1);	% log(p) & postprobs with old parameters
  nll = -sum(p);		% negative log-likelihood of data
  e = [e nll];
    
  if ((nll_old-nll) < tol*abs(nll_old)) || (n > maxit)
    gm.c = mu; gm.S = S; gm.p = pm; gm.type = cov_type;
    return;
  else
    nll_old = nll;
  end
    
  % M-step
  
  % Adjust the new estimates for the parameters
  new_p = sum(pm_x,1); new_c = pm_x'*X;
  
  % Re-estimate the mixing coefficients
  pm = new_p'/N;
  
  % Re-estimate the means
  mu = new_c./(new_p'*ones(1,D));
  
  % Re-estimate the covariances
  switch cov_type
   case 'i' 
    S = sum(sum(pm_x.*sqdist(X,mu),1),2)/(D*N); 
    % Ensure that no covariance is too small
    if (S < MIN_COV), S = init_cov; end      
   case 'I'
    S = (sum(pm_x.*sqdist(X,mu),1)./new_p)'/D;
    % Ensure that no covariance is too small      
    tmp = find(S < MIN_COV); S(tmp) = init_cov(tmp);
   case 'd'
    for m = 1:M
      diffs = bsxfun(@minus,X,mu(m,:));
      S(m,:) = sum(bsxfun(@times,diffs.*diffs,pm_x(:,m)),1);
    end
    S = sum(S,1)/N;
    % Ensure that no covariance is too small
    if (min(S) < MIN_COV), S = init_cov; end
   case 'D'
    for m = 1:M
      diffs = bsxfun(@minus,X,mu(m,:));        
      S(m,:) = sum(bsxfun(@times,diffs.*diffs,pm_x(:,m)),1)/new_p(m);
    end
    % Ensure that no covariance is too small      
    tmp = find(min(S,[],2)<MIN_COV); S(tmp,:) = init_cov(tmp,:);
   case 'F'
    for m = 1:M
      diffs = bsxfun(@times,bsxfun(@minus,X,mu(m,:)),sqrt(pm_x(:,m)));
      S(:,:,m) = (diffs'*diffs)/new_p(m);
    end
    % Ensure that no covariance is too small
    for m = 1:M        
      if (min(svd(S(:,:,m))) < MIN_COV), S(:,:,m) = init_cov(:,:,m); end
      % $$$ Alternative implementation
      % $$$ [U0,S0,V0] = svd(S(:,:,m));
      % $$$ S0 = diag(S0); S0(find(S0<MIN_COV)) = MIN_COV; S0 = diag(S0);
      % $$$ S(:,:,m) = U0*S0*V0';
    end
  end
end


function gm = GMinit(X,M,cov_type)	% Initialise GM by k-means

[N,D] = size(X); 

% Arbitrary width used to prevent variance collapse to zero: make it "large"
% so that each centre is responsible for a reasonable number of points
% (adapted from Netlab).
GMM_WIDTH = 1.0;

% Means: cluster means
[mu,labels] = kmeans(X,M);

% Indices of data points assigned to each cluster
I = cell(M,1); Il = zeros(M,1);
for m=1:M, I{m} = find(labels==m); Il(m) = length(I{m}); end

% Mixing proportions: fractions of data points assigned to each cluster
pm = Il/N;

% Covariance matrices: cluster covariances
switch cov_type
 case 'i'
  if M > 1
    % Determine widths as distance to nearest centre (or a constant if zero)
    cdist = sqdist(mu) + diag(sparse(ones(M,1)*realmax));
    S = min(cdist,[],1); S = S + GMM_WIDTH*(S < eps); S = min(S,[],2);
  else
    % Just use variance of all data points averaged over all dimensions
    S = mean(diag(cov(X)));
  end
 case 'I'
  if M > 1
    % Determine widths as distance to nearest centre (or a constant if zero)
    cdist = sqdist(mu) + diag(sparse(ones(M,1)*realmax));
    S = min(cdist,[],1)'; S = S + GMM_WIDTH*(S < eps);
  else
    % Just use variance of all data points averaged over all dimensions
    S = mean(diag(cov(X)));
  end
 case 'd'
  S = zeros(M,D); 
  for m = 1:M
    diffs = bsxfun(@minus,X(I{m},:),mu(m,:));     
    S(m,:) = sum((diffs.*diffs),1)/Il(m);
    % Replace small entries by GMM_WIDTH value
    S(m,:) = S(m,:) + GMM_WIDTH.*(S(m,:) < eps);
  end
  S = min(S,[],1);
 case 'D'
  S = zeros(M,D); 
  for m = 1:M
    diffs = bsxfun(@minus,X(I{m},:),mu(m,:));
    S(m,:) = sum((diffs.*diffs),1)/Il(m);
    % Replace small entries by GMM_WIDTH value
    S(m,:) = S(m,:) + GMM_WIDTH.*(S(m,:) < eps);
  end
 case 'F'
  S = zeros(D,D,M);
  for m = 1:M
    diffs = bsxfun(@minus,X(I{m},:),mu(m,:));
    S(:,:,m) = (diffs'*diffs)/Il(m);
    % Add GMM_WIDTH*Identity to rank-deficient covariance matrices
    if rank(S(:,:,m)) < D
      S(:,:,m) = S(:,:,m) + GMM_WIDTH.*eye(D);
    end
  end
end

gm.c = mu; gm.S = S; gm.p = pm; gm.type = cov_type;

