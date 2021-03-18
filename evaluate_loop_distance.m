skip = 200;
angle_list = [];
% % for i=skip+1:10:size(quat, 1)
% %     quat_i = quat(i,:);
% %     angle_min = 8;
% %     min_index = 1;
% %     for j=1:10:i-skip
% %         quater_dist = quaternion_mul_num(quat(j,:), [quat_i(1), -quat_i(2),-quat_i(3),-quat_i(4)]);
% %         axang = quat2axang(quater_dist);
% %         angle = abs(axang(4));
% %         if angle < angle_min
% %             angle_min = angle;
% %             min_index = j;
% %         end
% %     end
% %     angle_list = [angle_list;i min_index angle_min];
% % end

quat_zero = quat(skip, :);
for i=2*skip:10:size(quat,1)
    quater_dist = quaternion_mul_num(quat(i,:), [quat_zero(1), -quat_zero(2),-quat_zero(3),-quat_zero(4)]);
    axang = quat2axang(quater_dist);
    angle = abs(axang(4));
    angle_list = [angle_list;angle];
end
plot(angle_list);