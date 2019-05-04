%***********************************************************************************
%this code is created by Priyanka Goenka.
%This script is for image pre-processing of Extended Cohn-Kanade face image dataset.
%Here face portion of the images are cropped and saved in a new folder.
%***********************************************************************************

A=readtable('labels.csv'); %table with the image names and their labels of raw data.
%Replace 'C:\Users\Priyanka\Desktop\study...' with folder address of raw image data.
image_folder ='C:\Users\Priyanka\Desktop\study\MSU\2nd Semester\Pattern Recognition\Project\CKdataset\all-images';
filenames = dir(fullfile(image_folder, '*.png')); 
faceDetector = vision.CascadeObjectDetector; %Cascade object detector is used to detect faces in the image dataset.
%Shape inserter is used to insert shape around the face during face detection  
shapeInserter = vision.ShapeInserter('BorderColor','Custom','CustomBorderColor',[500]); 
B={};%Initialize an empty cell array B to store images which has labels. 
%Replace 'C:\Users\Priyanka\Desktop\study...' with folder address where the
%cropped images needs to be saved
Location_save='C:\Users\Priyanka\Desktop\study\MSU\2nd Semester\Pattern Recognition\Project\CKdataset\Cropped Images';
for i=1:size(A,1)
    if A{i,2} %2nd column of A is the image labels.  
        f= fullfile(image_folder, filenames(i).name); %to specify images names with full path and extension.
        our_images = imread(f); %to read images.
        BB = step(faceDetector, our_images); %BB is a rectangle which is drawn around the detected face portion.
       %crop the images
         for j = 1:size(BB,1)
             J= imcrop(our_images,BB(j,:));
         end
         baseFileName=filenames(i).name;%name of ith image data
         fullFileName = fullfile(Location_save, baseFileName);
         imwrite(J, fullFileName);% imread will save the images with the original names in the new location
         B=[B;(A(i,:))];% store the image info just like in A
     end
 end
 writetable(B,'ProcessedImageTable.csv')
