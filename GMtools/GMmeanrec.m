% [Xr,C] = GMmeanrec(X,M,gm)
% Missing data reconstruction from conditional means of a Gaussian mixture
%
% Given a data set X that contains missing values (indicated by the binary
% matrix M) and a Gaussian mixture density model gm, GMmeanrec reconstructs
% the missing values as the GM mean conditional on the present values, at
% each vector. Note that, unlike with GMmodesrec, each vector is reconstructed
% independently (no sequential structure assumed).
%
% In:
%   X: NxD data matrix with N D-dim data vectors stored rowwise.
%   M: NxD binary indicator matrix, where M(n,d) = 1 means that X(n,d) is
%      present (known value) while M(n,d) = 0 means missing (unknown value);
%      in the latter case, the value X(n,d) is ignored.
%   gm: Gaussian mixture on D variables (see GMpdf.m).
%
% Out:
%   Xr: NxD reconstructed data matrix.
%   C: Nx1 cell array, where entry n is an array containing the conditional
%      covariance of the missing values for vector X(n,:) (with dimension LxL
%      if there are L missing values at vector n).      

% Copyright (c) 2011 by Miguel A. Carreira-Perpinan and Chao Qin

function [Xr,C] = GMmeanrec(X,M,gm)

mu = gm.c; S = gm.S; pm = gm.p;		% Extract GM fields
[N,D] = size(X); Xr = X; C = cell(N,1);

for n=1:N
  I = find(M(n,:));			% indices for present variables
  J = setdiff(1:D,I);			% indices for missing variables  
  if ~isempty(J)			% some variables are missing
    o.P = I; o.M = J; o.xP = X(n,I);
    if nargout>=2
      [Xr(n,J),C{n}] = GMmoments(gm,o);	% mean and covariance of p(J|I)
    else
      Xr(n,J) = GMmoments(gm,o);	% mean of p(J|I)
    end
  end
end

