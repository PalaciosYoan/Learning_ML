% Copyright (c) 2016 by Miguel A. Carreira-Perpinan
% for use in CSE176 Introduction to Machine Learning at UC Merced

% This demonstrates how to use a Gaussian mixture (GM) model trained on a
% subset of MNIST digits in order to 1) do clustering and 2) reconstruct
% digit images with missing pixels.

load MNIST.mat; [N,D] = size(train_fea); DD = sqrt(D); Nt = size(test_fea,1);
train_fea = double(train_fea)/255; test_fea = double(test_fea)/255;

rng(1778);

% Subset to train and test on
%I1 = find(train_gnd==1);	% only 1s
I1 = randperm(N,1000); X1 = train_fea(I1,:); Y1 = train_gnd(I1);
I2 = randperm(Nt,10); X2 = test_fea(I2,:); Y2 = test_gnd(I2);

% Train a GM with K components
K = 10; tic; [gm,gm_e] = GMEM(X1,K,'F',[],[],10); toc

% Posterior probabilities of test set vectors
[p,~,pm_x] = GMpdf(X2,gm);			% pdf p(x) and postprobs p(m|x)

% Create a mask of missing components, for example:
Mr = rand(1,784) > 0.7;				% 70% missing pixels
Ml = [zeros(1,392) ones(1,392)];		% Left half missing
Mu = reshape(Ml,DD,DD); Mu = Mu'; Mu = Mu(:)';	% Upper half missing
% Reconstruct the test set vectors
M = Mu; tic; X2r = GMmeanrec(X2,repmat(M,size(X2,1),1),gm); toc


% ---------------------------------- Plots ----------------------------------

figure(1); plot(gm_e,'bo-');
xlabel('iteration'); ylabel('error'); title('EM learning curve');

% Plot the mean and weight of each GM component
figure(2); clf; colormap(gray(256));
K2 = ceil(sqrt(K)); K1 = ceil(K/K2);
for i=1:K
  subplot(K1,K2,i); imagesc(reshape(gm.c(i,:),DD,DD),[0 1]);
  set(gca,'XTick',[],'YTick',[]); axis image;
  title(['p(' num2str(i) ') = ' num2str(gm.p(i))]);
end

% Plot the posterior probabilities for the test set points
w = 0.05;				% width of image relative to [0,1]
AX = [0.5 K+0.5 0 1];
figure(3); clf; colormap(gray(256));
for i=1:size(X2,1)
  subplot(1,2,1); imagesc(reshape(X2(i,:),DD,DD),[0 1]);
  set(gca,'XTick',[],'YTick',[]); axis image;
  title(['Ground truth: ' num2str(Y2(i))]);
  subplot(1,2,2); bar(pm_x(i,:));
  axis(AX); xlabel('component index k'); ylabel('p(k|x)');
  P = get(gca,'Position');
  ax = P(3)/(AX(2)-AX(1)); bx = P(1) - ax*AX(1); % map coordinate -> figure
  ay = P(4)/(AX(4)-AX(3)); by = P(2) - ay*AX(3);
  for k=1:K
    axes('position',[ax*k+bx-w/2 ay*1.05+by-w/2 w w]);
    imagesc(reshape(gm.c(k,:),DD,DD),[0 1]); axis off; daspect([1 1 1]);
  end
  pause
end

% Plot the reconstruction results
figure(4); clf; colormap(gray(256));
for i=1:size(X2,1)
  subplot(1,3,1); imagesc(reshape(X2(i,:),DD,DD),[0 1]);
  set(gca,'XTick',[],'YTick',[]); axis image;
  title(['Ground truth: ' num2str(Y2(i))]);
  subplot(1,3,2); imagesc(reshape(X2r(i,:),DD,DD),[0 1]);
  set(gca,'XTick',[],'YTick',[]); axis image; title('Reconstruction');
  subplot(1,3,3); imagesc(reshape(M,DD,DD),[0 1]);
  set(gca,'XTick',[],'YTick',[]); axis image; title('Missing data mask');
  pause
end


% Suggestions of things to try:
% - Vary the proportion of missing data from 0 to 100%.
% - Vary the number of Gaussians (1, 10, 30).
% - Train only on digit 1s (for example).
% - Vary the size of the training set.
% - Train other types of GM (isotropic/diagonal/full covariances).
% - Create more types of masks (blocks, etc.).
% - Visualize the clusters using LDA (using the cluster assignments as labels).

