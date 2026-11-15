% FACEDETECTION  Stress-test three face detectors against progressive image degradation.
%
% For every image in each skin-tone dataset (Black / Brown / White), this script
% applies 7 degradation types at 10 increasing intensity steps and records whether
% each detector (Viola-Jones, CNN / Tiny YOLOv2, HOG + SVM) still finds a face.
% Results are written to results/<Method>/<Degradation>/<SkinTone>_Skintone_face_detection_results.csv
%
% Requirements: Image Processing Toolbox, Computer Vision Toolbox, Deep Learning Toolbox,
% Computer Vision Toolbox Model for YOLO v2 Object Detection, Statistics and Machine Learning Toolbox.

clc;
clear;
close all;

% Repository root (one level up from this file)
repoRoot = fileparts(fileparts(mfilename('fullpath')));

% Folder names for datasets
folders = {'Black', 'Brown', 'White'};

% Define base path (see data/README.md for how to download the dataset)
basePath = fullfile(repoRoot, 'data', 'train');
if ~isfolder(basePath)
    altBase = dir(fullfile(repoRoot, 'data', '**', 'train'));
    if ~isempty(altBase)
        basePath = fullfile(altBase(1).folder, altBase(1).name);
        fprintf('Base path auto-corrected to: %s\n', basePath);
    else
        error('Train folder not found. Download the dataset into data/train (see data/README.md).');
    end
end

% Output path
datafolder = fullfile(repoRoot, 'results');

% Load detectors
vjDetector = vision.CascadeObjectDetector();
cnnDetector = yolov2ObjectDetector('tiny-yolov2-coco');
load(fullfile(repoRoot, 'models', 'hogSVMClassifier.mat')); % loads hogSVMClassifier
imgSize = [64 64];

% Degradation setup
degradationTypes = {'Original', 'Decrease Brightness', 'Decrease Contrast', ...
                    'Salt & Pepper Noise', 'Gaussian Noise', 'Rotation', 'Blurred'};
numSteps = 10;
methods = {'ViolaJones', 'CNN', 'HOGSVM'};

header = [{'Image Name'}, arrayfun(@(k) sprintf('Increase %d', k), 0:numSteps-1, 'UniformOutput', false)];

% Define degradation functions (k = degradation step, 0..numSteps-1)
degradationFuncs = {
    @(img, k) img; % Original
    @(img, k) imadjust(img, [max(0, 0.5 + 0.05*k), 1], [0 1]); % Decrease Brightness
    @(img, k) imadjust(img, [max(0.4, 0.45 + 0.05*k), min(1, 0.55 + 0.05*k)], [0 1]); % Decrease Contrast
    @(img, k) imnoise(img, 'salt & pepper', 0.1 + 0.01*k); % Salt & Pepper Noise
    @(img, k) imnoise(img, 'gaussian', 0, 0.5 + 0.05*k); % Gaussian Noise
    @(img, k) imrotate(padarray(img, [30 30], 'replicate', 'both'), k * 30, 'crop'); % Rotation
    @(img, k) imgaussfilt(img, 2 + k); % Blurred
};

for f = 1:length(folders)
    folderName = folders{f};
    imagesFolder = fullfile(basePath, folderName);
    imds = imageDatastore(imagesFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

    imageLimit = min(500, numel(imds.Files));
    fprintf('\nProcessing %d images from %s...\n', imageLimit, folderName);

    % For saving nicer CSV names
    saveName = [folderName '_Skintone_face_detection_results.csv'];

    for m = 1:length(methods)
        method = methods{m};
        fprintf('\nUsing method: %s\n', method);

        for d = 1:length(degradationTypes)
            degradationType = degradationTypes{d};
            degradeFunc = degradationFuncs{d};

            % Folder for this method and degradation
            resultFolder = fullfile(datafolder, method, degradationType);
            if ~exist(resultFolder, 'dir')
                mkdir(resultFolder);
            end
            csvFilename = fullfile(resultFolder, saveName);

            % Initialise results
            results = cell(imageLimit + 1, numSteps + 1);
            results(1, :) = header;

            for i = 1:imageLimit
                img = readimage(imds, i);
                [~, name, ~] = fileparts(imds.Files{i});

                resultRow = {name};

                for step = 0:numSteps-1
                    degradedImg = degradeFunc(img, step);

                    % 3-channel RGB
                    if size(degradedImg, 3) == 1
                        degradedImg = cat(3, degradedImg, degradedImg, degradedImg);
                    end

                    grayImgTest = im2gray(degradedImg);

                    % Face detection methods
                    switch method
                        case 'ViolaJones'
                            bboxes = vjDetector(grayImgTest);
                            faceDetected = ~isempty(bboxes);

                        case 'CNN'
                            [bboxes, scores, labels] = detect(cnnDetector, degradedImg);
                            faceDetected = any(strcmp(labels, 'person')) || ~isempty(bboxes);

                        case 'HOGSVM'
                            resized = imresize(grayImgTest, imgSize);
                            features = extractHOGFeatures(resized);
                            prediction = predict(hogSVMClassifier, features);
                            faceDetected = strcmp(prediction, 'Face');
                    end

                    resultRow{end+1} = faceDetected;
                end

                results(i+1, :) = resultRow;

                fprintf('Method: %s | Folder: %s | Image %d/%d | Degradation: %s\n', ...
                        method, folderName, i, imageLimit, degradationType);
            end

            % Write the results table
            resultsTable = cell2table(results(2:end, :), 'VariableNames', results(1, :));
            try
                writetable(resultsTable, csvFilename);
            catch writeErr
                warning('Could not write to CSV: %s\nError: %s', csvFilename, writeErr.message);
            end
        end
    end
end

disp('All processing complete!');
