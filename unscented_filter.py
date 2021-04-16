import numpy as np
from math import exp, sqrt
from copy import deepcopy


def get_derivative(x):
    wb1=x[0];wb2=x[1];wb3=x[2]
    dqb1=x[3];dqb2=x[4];dqb3=x[5]
    k1 = x[6]
    k2 = x[7]
    p1 = exp(-k1)-exp(-k1-k2)
    p2 = exp(-k2)-exp(k1)
    p3 = exp(k1+k2)-exp(k2)
    dx = np.zeros((6,))
    dx[0]=p1*wb2*wb3
    dx[1]=p2*wb1*wb3
    dx[2]=p3*wb2*wb1
    wb_quat = np.asarray([0, wb1, wb2, wb3])
    dqb_quat = np.asarray([sqrt(1-dqb1**2-dqb2**2-dqb3**2), dqb1, dqb2, dqb3])
    dx[3:6] = 0.5*quaternion_mul_num(dqb_quat, wb_quat)[1:4]
    return dx

def ukf_f(x,dt):

    dx1 = get_derivative(x)
    x_n2 = deepcopy(x)
    x_n2[0:6] = x_n2[0:6]+0.5*dt*dx1
    dx2 = get_derivative(x_n2)
    x_n3 = deepcopy(x)
    x_n3[0:6] = x_n3[0:6]+0.5*dt*dx2
    dx3 = get_derivative(x_n3)
    x_n4 = deepcopy(x)
    x_n4[0:6] = x_n4[0:6]+0.5*dt*dx3
    dx4 = get_derivative(x_n4)
    
    x_n = deepcopy(x)
    x_n[0:6] = x_n[0:6] + dt/6*(dx1+2*dx2+2*dx3+dx4)

    return x_n

def ukf_h(x):
    dqb1=x[3];dqb2=x[4];dqb3=x[5]
    dqd1=x[8];dqd2=x[9];dqd3=x[10]
    h = np.zeros((3,))
    h[0] = dqb1 + dqd1 + dqb2*dqd3 - dqb3*dqd2
    h[1] = dqb2 + dqd2 + dqb3*dqd1 - dqb1*dqd3
    h[2] = dqb3 + dqd3 + dqb1*dqd2 - dqb2*dqd1
    return h

def quat_inv(q):
    new_q = np.zeros((4,))
    new_q[0] = q[0]
    new_q[1:4] = -q[1:4]
    return new_q

def quaternion_mul_num(q1, q2):
    q = np.zeros((4,))
    q[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    q[1] = q1[1]*q2[0] + q1[0]*q2[1] + q1[2]*q2[3] - q1[3]*q2[2]
    q[2] = q1[2]*q2[0] + q1[0]*q2[2] + q1[3]*q2[1] - q1[1]*q2[3]
    q[3] = q1[3]*q2[0] + q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1]
    return q

def ukf_update(UKF,y,x_global):
    M = UKF.x_prior
    dqb1=M[3];dqb2=M[4];dqb3=M[5]
    dqd1=M[8];dqd2=M[9];dqd3=M[10]
    qb = x_global[0:4]
    qd = x_global[4:8]
    # dqb_quat = np.asarray([sqrt(1-dqb1**2-dqb2**2-dqb3**2), dqb1, dqb2, dqb3])
    # dqd_quat = np.asarray([sqrt(1-dqd1**2-dqd2**2-dqd3**2), dqd1, dqd2, dqd3])
    # qb_new = quaternion_mul_num(qb, dqb_quat)
    # qd_new = quaternion_mul_num(dqd_quat, qd)

    delta_q = quaternion_mul_num(quat_inv(qb), y)
    delta_q = quaternion_mul_num(delta_q, quat_inv(qd))

    UKF.update(delta_q[1:4])
    M = UKF.x_post
    dqb1=M[3];dqb2=M[4];dqb3=M[5]
    dqd1=M[8];dqd2=M[9];dqd3=M[10]
    dqb_quat = np.asarray([sqrt(1-dqb1**2-dqb2**2-dqb3**2), dqb1, dqb2, dqb3])
    dqd_quat = np.asarray([sqrt(1-dqd1**2-dqd2**2-dqd3**2), dqd1, dqd2, dqd3])
    x_global[0:4] = quaternion_mul_num(qb, dqb_quat)
    x_global[4:8] = quaternion_mul_num(dqd_quat, qd)

    UKF.x[3:6] = 0
    UKF.x[8:11] = 0

    y_updated = quaternion_mul_num(x_global[0:4], x_global[4:8])

    return UKF, x_global, y_updated
