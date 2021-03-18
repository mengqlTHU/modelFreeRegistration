camera = pointCloud(calib(:,1:3));
robot = pointCloud(calib(:,4:6));
initialTrans = rigid3d([1 0 0;0 0 -1;0 1 0], [0 0 0.5]);
% [tform1, movingReg] = pcregistericp(camera,robot,'Transform', 'rigid',...
%                                     'InitialTransform', initialTrans);
[tform1, movingReg] = pcregistericp(camera,robot,'InitialTransform', initialTrans,...
                                    'Tolerance',[0.0001 0.0005],'MaxIterations',100);
pcshow(camera);
hold on
pcshow(robot);
T_rc = tform1.T;
T_rc'*[calib(:,1:3) ones(4,1)]'

