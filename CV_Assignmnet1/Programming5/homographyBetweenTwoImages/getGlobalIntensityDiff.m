function diff = getGlobalIntensityDiff(imgColor1, imgColor2)
    im1 = 0.299 * imgColor1(:,:,1) + 0.587 * imgColor1(:,:,2) + 0.114 * imgColor1(:,:,3);
    im2 = 0.299 * imgColor2(:,:,1) + 0.587 * imgColor2(:,:,2) + 0.114 * imgColor2(:,:,3);
    
    thresh = 5;   % Harris corner threshold
    nonmaxrad = 3;  % Non-maximal suppression radius
    %dmax = 200;
    w = 15;         % Window size for correlation matching
    sigma = 0.5;
    % Find Harris corners in image1 and image2
    [cim1, r1, c1] = harris(im1, sigma, thresh, nonmaxrad);
 
    [cim2, r2, c2] = harris(im2, sigma, thresh, nonmaxrad);
  
    [m1,m2] = matchbycorrelation(im1, [r1';c1'], im2, [r2';c2'], w);

    % Assemble homogeneous feature coordinates for fitting of the
    % homography, note that [x,y] corresponds to [col, row]
    x1 = [m1(2,:); m1(1,:); ones(1,length(m1))];
    x2 = [m2(2,:); m2(1,:); ones(1,length(m1))];    
    
    t = 0.005;  % Distance threshold for deciding outliers
    [H, inliers] = ransacfithomography(x1, x2, t);
  
    inliers1 = m1(:,inliers);
    inliers2 = m2(:,inliers);
    m1 = inliers1';
    m2 = inliers2';
    x = 1:length(m2);
    corr = [x' x'];
    %DisplayPoreCorr(im1, im2, m1, m2, corr);
    
    InverseOfH = inv(H); %when I calculate the transformed image, i use bilinear interpolation
    
    %at first, we need to calculate the coordinates for the four corners of
    %image1 in the coordinate system of img2
    [rowsIm1, colsIm1] = size(im1); 
    [rowsIm2, colsIm2] = size(im2);
    finalLeft = 1;
    finalRight = colsIm2;
    finalTop = 1;
    finalBot = rowsIm2;
    
    leftTopCornerCoord = H * [1;1;1];
    leftTopCornerCoord = leftTopCornerCoord / leftTopCornerCoord(3,1);
    if leftTopCornerCoord(1) < finalLeft
        finalLeft = floor(leftTopCornerCoord(1));
    end
    if leftTopCornerCoord(2) < finalTop
        finalTop = floor(leftTopCornerCoord(2));
    end
    
    RightTopCornerCoord = H * [colsIm1;1;1];
    RightTopCornerCoord = RightTopCornerCoord / RightTopCornerCoord(3,1);
    if RightTopCornerCoord(1) > finalRight
        finalRight = floor(RightTopCornerCoord(1));
    end
    if RightTopCornerCoord(2) < finalTop
        finalTop = floor(RightTopCornerCoord(2));
    end
    
    leftBotCornerCoord = H * [1;rowsIm1;1];
    leftBotCornerCoord = leftBotCornerCoord / leftBotCornerCoord(3,1);
    if leftBotCornerCoord(1) < finalLeft
        finalLeft = floor(leftBotCornerCoord(1));
    end
    if leftBotCornerCoord(2) > finalBot
        finalBot = floor(leftBotCornerCoord(2));
    end
    
    RightBotCornerCoord = H * [colsIm1;rowsIm1;1];
    RightBotCornerCoord = RightBotCornerCoord / RightBotCornerCoord(3,1);
    if RightBotCornerCoord(1) > finalRight
        finalRight = floor(RightBotCornerCoord(1));
    end
    if RightBotCornerCoord(2) > finalBot
        finalBot = floor(RightBotCornerCoord(2));
    end
    
    mergeRows = finalBot - finalTop + 1;
    mergeCols = finalRight - finalLeft + 1;
    transformedImage = zeros(mergeRows, mergeCols,3);
    transformedImage(-finalTop + 2 : -finalTop + 1 + rowsIm2, -finalLeft + 2 : -finalLeft + 1 + colsIm2,:) = imgColor2;
    %transformedImage(-finalTop + 3 : -finalTop + 2 + rowsIm2, -finalLeft + 2 : -finalLeft + 1 + colsIm2,:) = imgColor2;
    sumSharedRGBInImage2 = zeros(3,1);
    sumSharedRGBInImage1 = zeros(3,1);
    noOfSharedPixels = 0;
    for row = 1:mergeRows
        for col = 1: mergeCols
            oldPixel = 0;
            if transformedImage(row,col,1) ~=0
                oldPixel = 1;
                RGBImg2AtThisPos = [transformedImage(row,col,1);transformedImage(row,col,2);transformedImage(row,col,3)];
            end
            currentCoord = [col+finalLeft-1;row+finalTop-1;1];
            CoordInOriImage = InverseOfH * currentCoord;
            CoordInOriImage = CoordInOriImage / CoordInOriImage(3,1);
            
            xInSrcImage = CoordInOriImage(1,1);
            yInSrcImage = CoordInOriImage(2,1);
            
            floorY = floor(yInSrcImage);
            floorX = floor(xInSrcImage);
            ceilY = ceil(yInSrcImage);
            ceilX = ceil(xInSrcImage);
            normalizedX = xInSrcImage - floorX;
            normalizedY = yInSrcImage - floorY;
            
            if (floorX >= 1 && floorY >=1 && ceilX <= colsIm1 && ceilY <= rowsIm1) 
                f00 = imgColor1(floorY,floorX,1);
                f01 = imgColor1(ceilY,floorX,1);
                f10 = imgColor1(floorY,ceilX,1);
                f11 = imgColor1(ceilY,ceilX,1);
                transformedImage(row,col,1) = f00 + normalizedX * (f10 - f00)+ ...
                                            normalizedY * (f01 - f00) + ...
                                            normalizedX*normalizedY*(f00-f10-f01+f11);
                                        
                f00 = imgColor1(floorY,floorX,2);
                f01 = imgColor1(ceilY,floorX,2);
                f10 = imgColor1(floorY,ceilX,2);
                f11 = imgColor1(ceilY,ceilX,2);
                transformedImage(row,col,2) = f00 + normalizedX * (f10 - f00)+ ...
                                            normalizedY * (f01 - f00) + ...
                                            normalizedX*normalizedY*(f00-f10-f01+f11);
                                        
                f00 = imgColor1(floorY,floorX,3);
                f01 = imgColor1(ceilY,floorX,3);
                f10 = imgColor1(floorY,ceilX,3);
                f11 = imgColor1(ceilY,ceilX,3);
                transformedImage(row,col,3) = f00 + normalizedX * (f10 - f00)+ ...
                                            normalizedY * (f01 - f00) + ...
                                            normalizedX*normalizedY*(f00-f10-f01+f11);
                                        
                if oldPixel
                   noOfSharedPixels = noOfSharedPixels + 1;
                   sumSharedRGBInImage2 = sumSharedRGBInImage2 + RGBImg2AtThisPos;
                   sumSharedRGBInImage1 = sumSharedRGBInImage1 + [transformedImage(row,col,1);transformedImage(row,col,2);transformedImage(row,col,3)];
                end
            end
        end
    end
    averageSharedRGB1 = sumSharedRGBInImage1 / noOfSharedPixels;
    averageSharedRGB2 = sumSharedRGBInImage2 / noOfSharedPixels;
    diff = averageSharedRGB1 - averageSharedRGB2;
   