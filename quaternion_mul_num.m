% function q = quaternion_mul_num(qb, qd)
%     q = [0 0 0 0];
%     q(1) = qb(1)*qd(1) - qb(2)*qd(2) - qb(3)*qd(3) - qb(4)*qd(4);
%     q(2) = qb(2)*qd(1) + qb(1)*qd(2) + qb(4)*qd(3) - qb(3)*qd(4);
%     q(3) = qb(3)*qd(1) + qb(1)*qd(3) + qb(2)*qd(4) - qb(4)*qd(2);
%     q(4) = qb(4)*qd(1) + qb(1)*qd(4) + qb(3)*qd(2) - qb(2)*qd(3);
% end

function q = quaternion_mul_num(q1, q2)
    q = [0 0 0 0];
    q(1) = q1(1)*q2(1) - q1(2)*q2(2) - q1(3)*q2(3) - q1(4)*q2(4);
    q(2) = q1(2)*q2(1) + q1(1)*q2(2) + q1(3)*q2(4) - q1(4)*q2(3);
    q(3) = q1(3)*q2(1) + q1(1)*q2(3) + q1(4)*q2(2) - q1(2)*q2(4);
    q(4) = q1(4)*q2(1) + q1(1)*q2(4) + q1(2)*q2(3) - q1(3)*q2(2);
end