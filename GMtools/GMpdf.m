% [p,px_m,pm_x,pxm] = GMpdf(X,gm[,lg,o]) Gaussian mixture pdf
%
% Computes the value of p(x), p(x|m), p(m|x) and p(x,m) at each point in X,
% where p(x) is a Gaussian mixture with M components
%    p(x) = \sum^M_{m=1}{p(m).p(x|m)}
% where p(x|m) is a Gaussian distribution of mean c(m) and covariance S(m),
% coded in the struct gm (explained below).
% The struct array o allows marginalisation and conditioning, e.g.
% - o.P = [2 4], o.xP = [-1.2 2.3], o.M = [3]: p(x3|x2,x4 = [-1.2 2.3]).
% - o.P = [], o.M = [1 4]: p(x1,x4).
%
% Notes:
% - X must have D columns even if o.M has fewer than D variables.
% - If GMpdf runs out of memory, use a small enough number of points N.
% 
% In:
%   X: NxD matrix containing N D-dim points rowwise.
%   gm: GM mixture struct containing the following fields:
%      .c: MxD matrix containing the M D-dim centroids rowwise.
%          This sets the master values for M and D.
%      .S: covariances, coded as follows:
%       - S is DxDxM: full covariance, heteroscedastic ('F');
%       - S is DxD: full covariance, homoscedastic ('f');
%       - S is MxD: diagonal covariance, heteroscedastic ('D');
%       - S is 1xD: diagonal covariance, homoscedastic ('d');
%       - S is Mx1: isotropic covariance, heteroscedastic ('I');
%       - S is 1x1: isotropic covariance, homoscedastic ('i').
%      .p: Mx1 list containing the mixing proportions.
%      .type: one of the chars above ('F','f','D','d','I','i').
%       This field should agree with the result of GMtype(gm).
%   lg: if 1, return log(p),log(px_m),log(pxm) rather than p,px_m,pxm.
%      If 0, return p,px_m,pxm. Default: 0.
%      The log values are useful to avoid underflow, when p,px_m,pxm are very
%      small (this can often happen when the dimension of X is high).
%   o: struct array containing the following fields:
%      .P: subset of 1..D indicating what variables are present, i.e.,
%          what variables we condition on.
%      .xP: 1x? vector containing the values of the present variables.
%      .M: subset of 1..D indicating what variables are missing, i.e.,
%          neither present nor marginalised over.
%      Default: o.P = [], o.M = 1..D.
% Out:
%   p: Nx1 list of values of the probability density p(x) at X.
%   px_m: NxM list of values of the forward probability p(x|m) at X.
%   pm_x: NxM list of values of the posterior probability p(m|x) at X.
%   pxm: NxM list of values of the joint probability p(x,m) at X.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.
%
% Copyright (c) 2016 by Miguel A. Carreira-Perpinan and Chao Qin

function [p,px_m,pm_x,pxm] = GMpdf(X,gm,lg,o)

% ---------- Argument defaults ----------
if ~exist('lg','var') || isempty(lg) lg = 0; end;
if exist('o','var') && ~isempty(o)
  % Transform parameters, then call the function again without "o"
  switch nargout
   case 1, p = GMpdf(X(:,o.M),GMcondmarg(gm,o),lg);
   case 2, [p,px_m] = GMpdf(X(:,o.M),GMcondmarg(gm,o),lg);
   case 3, [p,px_m,pm_x] = GMpdf(X(:,o.M),GMcondmarg(gm,o),lg);
   otherwise, [p,px_m,pm_x,pxm] = GMpdf(X(:,o.M),GMcondmarg(gm,o),lg);
  end
  return;
end
% ---------- End of "argument defaults" ----------

mu = gm.c; S = gm.S; pm = gm.p; cov_type = gm.type;	% Extract GM fields
[M,D] = size(mu); N = size(X,1);

% The computations below are numerically accurate in the following sense:
% - p(m|x): always accurate, even if, for a given x, p(x|m) underflows for
%   each m. We divide the numerator and denominator of p(m|x) by the largest
%   entry.
% - p(x),p(x|m),p(x,m): with lg=1 we return the log of these values, which is
%   always accurate even if p(x|m) underflows for each m. With lg=0, if p(x|m)
%   underflows for each m then p(x),p(x|m),p(x,m) will equal 0 for all m.

% Compute the argument of the exponentials (including part of the normalization
% constant):
switch cov_type
 case 'F'
  argexp = zeros(N,M); logz = zeros(M,1);
  for m = 1:M
    diffs = bsxfun(@minus,X,mu(m,:));    
    % Use spectral decomposition of the covariance matrix to speed computation
    [U,L] = eig(S(:,:,m)); L = diag(L); logz(m) = -sum(log(L))/2;
    temp = diffs*U*diag(sparse((2*L).^(-1/2))); argexp(:,m) = -sum(temp.^2,2);
  end  
 case 'f'
  argexp = zeros(N,M);
  [U,L] = eig(S); L = diag(L); logz = repmat(-sum(log(L))/2,M,1);
  for m = 1:M
    diffs = bsxfun(@minus,X,mu(m,:));    
    temp = diffs*U*diag(sparse((2*L).^(-1/2))); argexp(:,m) = -sum(temp.^2,2);
  end  
 case 'i'
  argexp = -sqdist(X,mu)/(2*S); logz = repmat(-D*log(S)/2,M,1);
 case 'I'
  argexp = -sqdist(X,mu)*diag(sparse((2*S).^(-1))); logz = -D*log(S)/2;
 case 'd'  
  SS = diag(sparse(sqrt(2*S))); argexp = -sqdist(X/SS,mu/SS);
  logz = repmat(-sum(log(S))/2,M,1);
 case 'D'
  argexp = zeros(N,M); logz = -sum(log(S),2)/2;
  for m = 1:M
    diffs = bsxfun(@minus,X,mu(m,:));
    argexp(:,m) = -sum(diffs.^2/diag(sparse(2*S(m,:))),2);
  end
end
argexp = bsxfun(@plus,argexp,logz');

% p(x) and p(x|m) or their log values
if lg			% correct argexp by subtracting the max over components
  px_m = -D/2*log(2*pi) + argexp;
  argexp = bsxfun(@plus,argexp,log(pm)');
  L = max(argexp,[],2); argexp = exp(bsxfun(@minus,argexp,L));
  p = -D/2*log(2*pi) + L + log(sum(argexp,2));
else
  px_m = (2*pi)^(-D/2)*exp(argexp); p = px_m*pm;
end

% p(m|x)
if nargout>=3
  if ~lg		% correct argexp
    argexp = bsxfun(@plus,argexp,log(pm)');
    pm_x = exp(bsxfun(@minus,argexp,max(argexp,[],2)));
    pm_x = bsxfun(@rdivide,pm_x,sum(pm_x,2));
  else			% argexp corrected in p(x) computation
    pm_x = bsxfun(@rdivide,argexp,sum(argexp,2));
  end
end

% p(x,m) or its log value
if nargout>=4
  if lg
    pxm = bsxfun(@plus,px_m,log(pm'));
  else
    pxm = bsxfun(@times,px_m,pm');
  end
end

