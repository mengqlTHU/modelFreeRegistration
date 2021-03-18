function w = get_w_in_body_frame(q1, q2, dt)
  q_dot = (q2-q1)/dt;
  w1 = quaternion_mul_num([q1(1) -q1(2) -q1(3) -q1(4)],2*q_dot);
  w2 = quaternion_mul_num([q2(1) -q2(2) -q2(3) -q2(4)],2*q_dot);
  w = (w1+w2)/2;
  w = w(2:4);
end

