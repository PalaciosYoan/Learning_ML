% Copyright (c) 2016 by Miguel A. Carreira-Perpinan
% for use in CSE176 Introduction to Machine Learning at UC Merced

% Plot stuff from the MNIST dataset:
% handwritten digit images of 28x28 pixels as row vectors of 1x784

load MNIST.mat; [N,D] = size(train_fea); DD = sqrt(D);

% Let us work with a random subset of it
I = randperm(N,1000);
X = double(train_fea(I,:))/255;		% pixel values as vectors of 1xD
Y = double(train_gnd(I)); Y(Y==10)=0;	% class label in 0-9

% Plot a subset of vectors (images)
figure(1); clf; colormap(gray(256)); K1 = 5; K2 = 6;
for i=1:K1*K2
  subplot(K1,K2,i); imagesc(reshape(X(i,:),DD,DD),[0 1]);
  set(gca,'XTick',[],'YTick',[]); axis image;
end

% Plot several statistics

% Ranges of values: classes should be 0-9, pixel values should be in [0,1]
unique(Y)'
minX = min(X,[],1); maxX = max(X,[],1);

% Mean and covariance matrix
meanX = mean(X,1); covX = cov(X);

% Plot as vectors...
figure(2); clf; plot(1:D,minX,'b-',1:D,meanX,'k-',1:D,maxX,'r-');
axis([0 D+1 -0.05 1.05]); xlabel('pixel index'); legend('min','mean','max');
% ...or as images
figure(3); clf; colormap(gray(256));
subplot(1,3,1); imagesc(reshape(minX,DD,DD),[0 1]);
set(gca,'XTick',[],'YTick',[]); axis image;
title({'minimum value','at each pixel'});
subplot(1,3,2); imagesc(reshape(maxX,DD,DD),[0 1]);
set(gca,'XTick',[],'YTick',[]); axis image;
title({'maximum value','at each pixel'});
subplot(1,3,3); imagesc(reshape(meanX,DD,DD),[0 1]);
set(gca,'XTick',[],'YTick',[]); axis image;
title({'mean value','at each pixel'});

% Plot covariance matrix (negative and positive values)
figure(4); clf; colormap(parula(256)); imagesc(covX); colorbar;
set(gca,'XTick',[],'YTick',[]); axis image; title('covariance matrix');
% Diagonal of the covariance matrix = variance of each pixel
figure(5); clf; plot(1:D,diag(covX),'b-');		% as vector
xlim([0 D+1]); xlabel('pixel index'); ylabel('variance');
figure(6); clf; colormap(gray(256));
imagesc(reshape(diag(covX),DD,DD));			% as img
set(gca,'XTick',[],'YTick',[]); axis image; title('pixel variance');
% Covariance of the central pixel wrt the other pixels
idx = [DD/2 DD/2];					% pixel index
IDX = sub2ind([DD DD],idx(1),idx(2));
figure(7); clf; plot(1:D,covX(IDX,:),'b-');		% as vector
xlim([0 D+1]); xlabel('pixel index');
ylabel(['covariance for pixel ' num2str(idx)]);
figure(8); clf; colormap(parula(256));
imagesc(reshape(covX(IDX,:),DD,DD)); colorbar;		% as img
hold on; plot(idx(2),idx(1),'k*'); hold off;
set(gca,'XTick',[],'YTick',[]); axis image;
title(['covariance for pixel ' num2str(idx)]);

% Plot the mean of each digit type
figure(9); clf; colormap(gray(256));
digits = unique(Y)';
for k = digits
  z = find(Y==k); m = mean(X(z,:)); C = cov(X(z,:));
  subplot(1,max(digits)+1,k+1); imagesc(reshape(m,DD,DD),[0 1]);
  set(gca,'XTick',[],'YTick',[]); axis image; title(['mean(' num2str(k) 's)']);
end

% Plot the histogram of digit classes
figure(10); clf; hist(Y,0:9); xlim([-1 10]);
xlabel('digit class'); ylabel('frequency');

% Plot the histogram of a single pixel
idx = [DD/2 DD/2];					% pixel index
IDX = sub2ind([DD DD],idx(1),idx(2));
figure(11); clf; hist(X(:,IDX),linspace(0,1,10));
xlabel('pixel value (grayscale)'); ylabel('frequency');
title(['pixel ' num2str(idx)]);

% Plot the histogram of a pair of pixels
idx1 = [DD/2 DD/2]; idx2 = [DD/2+1 DD/2];		% pixel indices
IDX1 = sub2ind([DD DD],idx1(1),idx1(2));
IDX2 = sub2ind([DD DD],idx2(1),idx2(2));
figure(12); clf; hist3(X(:,[IDX1 IDX2]),{linspace(0,1,10),linspace(0,1,10)});
xlabel('pixel value (grayscale)'); ylabel('pixel value (grayscale)');
zlabel('frequency');
title(['pixels ' num2str(idx1) ' and ' num2str(idx2)]);

% Scatterplots of pairs of pixels
idx1 = [DD/2 DD/2]; idx2 = [DD/2+1 DD/2];		% pixel indices
IDX1 = sub2ind([DD DD],idx1(1),idx1(2));
IDX2 = sub2ind([DD DD],idx2(1),idx2(2));
figure(13); clf; plot(X(:,IDX1),X(:,IDX2),'b.');
xlabel(['pixel ' num2str(idx1)]); ylabel(['pixel ' num2str(idx2)]);
daspect([1 1 1]); axis([-0.1 1.1 -0.1 1.1]);


% Suggestions of things to try:
% - Plot the covariance for only the digit 1s, for example.
% - Plot the histogram of other pixels (e.g. in the corners of the image).
% - Plot scatterplots of other pairs of pixels (neighboring or far apart).
% - Plot scatterplots for pairs of pixels only for the digit 1s.

