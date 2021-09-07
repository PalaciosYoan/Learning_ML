% Copyright (c) 2016 by Miguel A. Carreira-Perpinan
% for use in CSE176 Introduction to Machine Learning at UC Merced

% Dataset for 2D clustering or classification: several Gaussian clusters.
% We use the GMtools for simplicity. You can also generate each cluster
% using randn and applying a linear transformation to the result.

rng(0);		% seed the random number generator for repeatability

% Parameters for a Gaussian mixture with 4 components in 2D:
% - Mixing proportion (prior probability) of each component, gm.p
%   (how many points correspond to it).
% - Mean of each component, gm.c
%   (where it is located).
% - Covariance matrix of each component, gm.S.
%   (what shape it has and how big is its spread).
gm.p = [2 2 1 5]'; gm.p = gm.p/sum(gm.p); gm.c = [0 1;-2 5;-3 4;7 5];
gm.S(:,:,1) = [1 0;0 2]; gm.S(:,:,2) = [2 1;1 1];
gm.S(:,:,3) = [0.2 0.1;0.1 1]; gm.S(:,:,4) = [3 -1;-1 2]; gm.type = 'F';

% Sample 350 points in 2D with GMsample
N = 350; Y = GMsample(N,gm);

% Plot
figure(1); plot(Y(:,1),Y(:,2),'b+');
xlabel('x_1'); xlabel('x_2'); daspect([1 1 1]);

hold on;
% Plot mean of each component
text(gm.c(:,1),gm.c(:,2),num2str([1:length(gm.p)]'),'FontSize',24,'Color','r');
plot(gm.c(:,1),gm.c(:,2),'ro');
% Plot covariance matrix for each component as one standard deviation contour
t = 0:2*pi/1000:2*pi; xy = [cos(t)' sin(t)']; xx = [0 1;-1 0;0 -1;1 0];
for m = 1:length(gm.p)
  switch gm.type
   case 'F', [V,D] = eig(gm.S(:,:,m));
   case 'D', [V,D] = eig(diag(gm.S(m,:)));
   case 'd', [V,D] = eig(diag(gm.S));
   case 'I', [V,D] = eig(gm.S(m)*eye(size(Y,2)));
   case 'i', [V,D] = eig(gm.S*eye(size(Y,2)));
  end
  mu = gm.c(m,:);
  xy_new = bsxfun(@plus,xy*(V*sqrt(D))',mu);
  yy = bsxfun(@plus,xx*(V*sqrt(D))',mu);
  plot([yy(1,1) yy(3,1)],[yy(1,2) yy(3,2)],'r-',...
       [yy(2,1) yy(4,1)],[yy(2,2) yy(4,2)],'r-',...
       xy_new(:,1),xy_new(:,2),'r-','LineWidth',2);
end
hold off;


% Suggestions of things to try:
% - Change the number of points N.
% - Change the parameters of the Gaussian components: mixing proportions,
%   means, covariances.

