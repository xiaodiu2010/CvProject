compile;

% load and display model
load('PARSE_final');

pose_result = zeros(1004,107);
imlist = dir('../database/photos/*.jpg');
for i = 1:length(imlist)
    % load and display image
    im = imread(['../database/photos/' imlist(i).name]);
    clf; imagesc(im); axis image; axis off; drawnow;
    num = strsplit(imlist(i).name,'.');
    num = num{1};
    pose_result(i,1) = str2double(num);
    % call detect function
    tic;
    boxes = detect(im, model, min(model.thresh,-1));
    dettime = toc; % record cpu time
    boxes = nms(boxes, .1); % nonmaximal suppression
    colorset = {'g','g','y','m','m','m','m','y','y','y','r','r','r','r','y','c','c','c','c','y','y','y','b','b','b','b'};
    showboxes(im, boxes(1,:),colorset); % show the best gt
    %showboxes(im, boxes,colorset);  % show all detections
    saveas(gcf,['../database/poseresult/' imlist(i).name])
    pose_result(i,2:107) = boxes(1,:);
    close all
    fprintf('detection took %.1f seconds\n',dettime);
    disp('press any key to continue');
    pause;
end

save('../database/pose_result.mat',pose_result);
disp('done');
