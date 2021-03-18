omega_body = zeros(size(omega_inertial));
% for i=1:size(omega_body,1)
%     R_for_quat = quat2rotm(quat(i,:));
%     omega_body(i,:) = (R_for_quat'*omega_inertial(i,:)')';
% end
for i=1:size(omega_body,1)-1
    omega_body(i, :) = get_w_in_body_frame(quat(i,:), quat(i+1,:), 0.05);
end