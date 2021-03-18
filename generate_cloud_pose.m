filepath = "20200205_3axis";
matlab_log = fopen(strcat(filepath,"/","log_matlab.txt"),'r');
cpp_log = fopen(strcat(filepath,"/","timestamp.txt"),'r');

matlab_jp = cell2mat(textscan(matlab_log, "%f64,%f64,%f64,%f64,%f64,%f64,%f64\n"));
cpp_time = cell2mat(textscan(cpp_log, "%f,%f64"));

result = zeros(size(cpp_time, 1), 10);
cur = 1;
tool_form = eye(4);
tool_form(3,4) = 110;
first_quat = quat(1,:);
for i=1:size(cpp_time,1)
    t = cpp_time(i,2);
    while cur < size(matlab_jp,1) && t>matlab_jp(cur, 1)
        cur = cur + 1;
    end
    jp = matlab_jp(cur, 2:7);
    tool_pose = UR_fkin('UR3',jp,tool_form);
    tool_pose(1:3,4) = tool_pose(1:3,4)/1000;
    T_ct = inv(T_rc')*tool_pose;
    quat_ct = quaternion_mul_num(tform2quat(T_ct), first_quat);
    euler_ct = tform2eul(T_ct, 'ZYX');
    pos_ct = T_ct(1:3,4)';
    result(i,1:3) = pos_ct;
    result(i,4:6) = euler_ct;
    result(i,7:10) = quat_ct;
end
writematrix(result, strcat(filepath,"/","point_cloud_pose.txt"));