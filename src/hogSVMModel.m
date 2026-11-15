% HOGSVMMODEL  Train the custom HOG + SVM face / non-face classifier.
%
% Uses a separate face and non-face dataset (Caltech Faces and Non Faces, Kaggle)
% so the classifier never sees the images it is later evaluated on.
% Output: models/hogSVMClassifier.mat (variable: hogSVMClassifier)

clc;
clear;

% Repository root (one level up from this file)
repoRoot = fileparts(fileparts(mfilename('fullpath')));

% Training image folders (see data/README.md)
facesPath = fullfile(repoRoot, 'data', 'Faces', 'faces');
nonFacesPath = fullfile(repoRoot, 'data', 'Faces', 'nonfaces');

% Load image datastores
faceDataset = imageDatastore(facesPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
nonFaceDataset = imageDatastore(nonFacesPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Set image size for HOG feature extraction
imgSize = [64 64];

% Define HOG feature extraction function
extractHOG = @(img) extractHOGFeatures(imresize(im2gray(img), imgSize));

% Extract HOG features from face images
numFaces = numel(faceDataset.Files);
faceFeatures = zeros(numFaces, 1764);
for i = 1:numFaces
    img = readimage(faceDataset, i);
    faceFeatures(i, :) = extractHOG(img);
end

% Extract HOG features from non-face images
numNonFaces = numel(nonFaceDataset.Files);
nonFaceFeatures = zeros(numNonFaces, 1764);
for i = 1:numNonFaces
    img = readimage(nonFaceDataset, i);
    nonFaceFeatures(i, :) = extractHOG(img);
end

% Combine features and create labels
features = [faceFeatures; nonFaceFeatures];
labels = [repmat("Face", numFaces, 1); repmat("NonFace", numNonFaces, 1)];

% Train SVM classifier
hogSVMClassifier = fitcsvm(features, labels);

% Save the trained model
save(fullfile(repoRoot, 'models', 'hogSVMClassifier.mat'), 'hogSVMClassifier');

disp('HOG + SVM model trained and saved successfully.');
