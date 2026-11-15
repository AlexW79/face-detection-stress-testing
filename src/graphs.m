% GRAPHS  Plot detection counts per degradation step for each method / skin tone.
%
% Reads the CSVs written by facedetection.m from results/ and draws, for every
% detection method and degradation type, the number of images (out of 500) in
% which a face was detected at each degradation step, one line per skin tone.

clc;
clear;
close all;

% Paths and setup
repoRoot = fileparts(fileparts(mfilename('fullpath')));
datafolder = fullfile(repoRoot, 'results');
methods = {'ViolaJones', 'CNN', 'HOGSVM'};
folders = {'Black', 'Brown', 'White'};
degradationTypes = {'Original', 'Decrease Brightness', 'Decrease Contrast', ...
                    'Salt & Pepper Noise', 'Gaussian Noise', 'Rotation', 'Blurred'};
numSteps = 10;

for m = 1:length(methods)
    method = methods{m};
    figure('Name', method);
    tiledlayout('flow', 'Padding', 'compact');

    for j = 1:length(degradationTypes)
        degradationType = degradationTypes{j};
        nexttile;
        hold on;

        for f = 1:length(folders)
            folderName = folders{f};
            csvFile = fullfile(datafolder, method, degradationType, ...
                               [folderName '_Skintone_face_detection_results.csv']);

            if ~exist(csvFile, 'file')
                warning('Missing CSV: %s', csvFile);
                continue;
            end

            T = readtable(csvFile, 'VariableNamingRule', 'preserve');
            totalDetected = zeros(1, numSteps);
            for step = 1:numSteps
                detections = T.(sprintf('Increase %d', step-1));
                if iscell(detections)
                    detections = cellfun(@(x) isequal(x,true) || isequal(x,1), detections);
                end
                totalDetected(step) = sum(detections);
            end

            plot(0:numSteps-1, totalDetected, '-o', 'DisplayName', folderName);
        end

        title(degradationType);
        xlabel('Degradation Level');
        ylabel('Images Detected');
        legend('Location', 'best');
        grid on;
    end

    sgtitle(['Face Detection Robustness - ' method]);
end
