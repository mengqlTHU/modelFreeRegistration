function qd = find_nearest_quat(qd_star,qd_target)
dist = [];
for theta = 0:0.001:2*pi
    rot_quat = [cos(theta/2),0,0,sin(theta/2)];
    qd_rot = quaternion_mul_num(rot_quat, qd_star);
    qd_diff = quaternion_mul_num([qd_rot(1),-qd_rot(2),-qd_rot(3),-qd_rot(4)], qd_target);
    axang_diff = quat2axang(qd_diff);
    dist= [dist;theta,axang_diff(4)];
end
plot(dist(:,1),dist(:,2));
qd = qd_star;
end

