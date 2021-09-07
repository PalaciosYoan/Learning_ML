% [Xr,modes,pmodes] = GMmodesrec(X,M,gm[,w,tol,maxit,mindiff,maxeig,th])
% Sequential data reconstruction from conditional modes of a Gaussian mixture
%
% Given a data set X where data vectors 1..N are sequential that contains
% missing values (indicated by the binary matrix M) and a Gaussian mixture
% density model gm, GMmodesrec reconstructs the missing values as as smooth
% trajectory through the set of modes of the GM conditional on the present
% values, at each vector.
%
% In:
%   X,M,gm: see GMmeanrec.
%   w: 1xD vector containing the weights for the weighted Euclidean
%      distance (default: ones, i.e., unweighted Euclidean distance).
%   tol,maxit,mindiff,maxeig,th: see GMmodes.
%
% Out:
%   Xr: NxD reconstructed data matrix.
%   modes: Nx1 cell array, where entry n is an array containing rowwise all
%      the modes of the conditional distribution for vector X(n,:) of missing
%      given present values.
%   pmodes: like modes but containing the modes' pdf value.

% Copyright (c) 2011 by Miguel A. Carreira-Perpinan and Chao Qin

function [Xr,modes,pmodes] = GMmodesrec(X,M,gm,w,varargin)

% ---------------- Argument defaults ---------------
if ~exist('w','var') || isempty(w) w = ones(1,size(X,2)); end;
% ---------- End of "argument defaults" ------------

mu = gm.c; S = gm.S; pm = gm.p;		% extract GM fields
[N,D] = size(X); modes = cell(N,1); pmodes = cell(N,1);

for n=1:N
  I = find(M(n,:));			% indices for present variables
  J = setdiff(1:D,I);			% indices for missing variables  
  if isempty(J)				% no missing variables
    modes{n} = X(n,I); pmodes{n} = 0;
  else
    o.P = I; o.M = J; o.xP = X(n,I);
    if isempty(I)			% no present variables
      % Computing all the modes of p(x) is slow so we cache it in Ym
      if ~exist('Ym','var') [Ym,pYm] = GMmodes(gm,[],varargin{:}); end;
      temp = Ym; pmodes{n} = pYm;	% precomputed
    else			% some variables are missing, some present
      [temp,pmodes{n}] = GMmodes(gm,o,varargin{:});	% all modes of p(J|I)
    end
    nmodes = length(pmodes{n}); modes{n} = zeros(nmodes,D);
    modes{n}(:,I) = repmat(X(n,I),nmodes,1); modes{n}(:,J) = temp;
  end
end

% Greedy and dynamic programming searches (with optional weights)
Xr = shp_dp(modes,{[],I},w);		% dynamic programming search
%Xr = shp_dp(modes,{[],I},w,pmodes,[0 1]);	% with mode weights

