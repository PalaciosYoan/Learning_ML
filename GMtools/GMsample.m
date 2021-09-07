% [X,Y,pX] = GMsample(N,gm[,o]) Samples from a Gaussian mixture
%
% In:
%   N: number of samples to generate.
%   gm,o: see GMpdf.m
% Out:
%   X: NxD matrix containing the samples rowwise.
%   Y: Nx1 vector containing the index of the component that generated X(n).
%   pX: Nx1 vector containing the values of p(x) at the samples.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.
%
% Copyright (c) 2016 by Miguel A. Carreira-Perpinan and Chao Qin

function [X,Y,pX] = GMsample(N,gm,o)

% ---------- Argument defaults ----------
if exist('o','var') && ~isempty(o)
  % Transform the parameters, then call the function again without "o"
  switch nargout
   case 1, X = GMsample(N,GMcondmarg(gm,o));
   case 2, [X,Y] = GMsample(N,GMcondmarg(gm,o));
   otherwise, [X,Y,pX] = GMsample(N,GMcondmarg(gm,o));
  end
  return;
end
% ---------- End of "argument defaults" ----------

mu = gm.c; S = gm.S; pm = gm.p; cov_type = gm.type;	% Extract GM fields
[M,D] = size(mu);

% Generate N samples from a discrete distribution corresponding to pm and
% create its histogram of counts, Np.
u = rand(N,1); Np = zeros(size(pm));
accp = cumsum(pm); accp(M) = 1.1;	% To ensure that u(n)<accp(M)
for n=1:N
  c = M + 1 - sum(u(n)<accp); Np(c) = Np(c) + 1;
end

% Now Np(m) contains the number of samples to be generated from component m.
X = []; Y = zeros(N,1); idx = 1;
for m=1:M
  if Np(m)>0
    Y(idx:idx+Np(m)-1) = m; idx = idx + Np(m);
    Q = randn(Np(m),D);
    switch cov_type
     case 'F'
      [U,L] = eig(S(:,:,m));
      x = bsxfun(@plus,Q*bsxfun(@times,sqrt(diag(L)),U'),mu(m,:));
     case 'i', x = bsxfun(@plus,sqrt(S)*Q,mu(m,:));
     case 'I', x = bsxfun(@plus,sqrt(S(m))*Q,mu(m,:));
     case 'd', x = bsxfun(@plus,bsxfun(@times,Q,sqrt(S)),mu(m,:));
     case 'D', x = bsxfun(@plus,bsxfun(@times,Q,sqrt(S(m,:))),mu(m,:));
    end
    X = [X; x];
  end
end

if nargout>2 pX = GMpdf(X,gm); end;

