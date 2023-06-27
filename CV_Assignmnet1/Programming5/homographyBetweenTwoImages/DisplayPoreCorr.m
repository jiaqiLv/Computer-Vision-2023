function DisplayPoreCorr(im1, im2, pores1, pores2, corr)
 % Input: 
  % im1 im2: two input images
  % pores1,pores2: 2D coordinates of points in im1 im2 
  % corr:  correspondences indexes between pores1 and pores2
[rows1, cols1] = size(im1);
[rows2, cols2] = size(im2);
 
rows = max([rows1, rows2]);
cols = cols1 + cols2 + 3;
im = zeros(rows, cols);
mask = zeros(rows, cols);
 
im(1:rows1, 1:cols1) = im1;
im(1:rows2, cols1+4:cols) = im2;
 
% pp1 = pores1(corr(:, 1), end-3:end-2);
% pp2 = pores2(corr(:, 2), end-3:end-2);
pp1 = pores1(corr(:, 1), 1:2);
pp2 = pores2(corr(:, 2), 1:2);
 
pp2(:, 2) = pp2(:, 2) + cols1 + 3;
 
figure;imshow(im,[]);
hold on
for i = 1:size(corr, 1)
    y1 = pp1(i, 1);
    x1 = pp1(i, 2);
    plot(x1, y1, 'Marker', 'o', 'MarkerEdgeColor', 'r', 'MarkerSize', 5);
    xt = x1;
    yt = y1;
    while mask(yt, xt) ~= 0
        if yt < rows - 6
            yt = yt + 6;
        else
            yt = yt - 6;
        end
    end
    %text(xt, yt, num2str(i), 'color', 'b');
    mask(yt, xt) = 1;
    y2 = pp2(i, 1);
    x2 = pp2(i, 2);
    plot(x2, y2, 'Marker', 'o', 'MarkerEdgeColor', 'r', 'MarkerSize', 5);
    xt = x2;
    yt = y2;
    while mask(yt, xt) ~= 0
        if yt < rows - 6
            yt = yt + 6;
        else
            yt = yt - 6;
        end
    end
    %text(xt, yt, num2str(i), 'color', 'b');
    mask(yt, xt) = 1;
    line([x1, x2], [y1, y2], 'Color', 'y');
end
hold off
