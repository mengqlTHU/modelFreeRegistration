import numpy as np
from math import exp
from copy import deepcopy

def get_derivative(x):
    wb1=x[0];wb2=x[1];wb3=x[2]
    qb0=x[3];qb1=x[4];qb2=x[5];qb3=x[6]
    k1 = x[7]
    k2 = x[8]
    p1 = exp(-k1)-exp(-k1-k2)
    p2 = exp(-k2)-exp(k1)
    p3 = exp(k1+k2)-exp(k2)
    dx = np.zeros((7,))
    dx[0]=p1*wb2*wb3
    dx[1]=p2*wb1*wb3
    dx[2]=p3*wb2*wb1
    dx[3]=0.5*(-wb1*qb1-wb2*qb2-wb3*qb3)
    dx[4]=0.5*(wb1*qb0-wb2*qb3+wb3*qb2)
    dx[5]=0.5*(wb1*qb3+wb2*qb0-wb3*qb1)
    dx[6]=0.5*(-wb1*qb2+wb2*qb1+wb3*qb0)
    return dx

def ekf_df_dx(x,dt):

    wb1=x[0];wb2=x[1];wb3=x[2]
    qb0=x[3];qb1=x[4];qb2=x[5];qb3=x[6]
    k1 = x[7]
    k2 = x[8]
    p1 = exp(-k1)-exp(-k1-k2)
    p2 = exp(-k2)-exp(k1)
    p3 = exp(k1+k2)-exp(k2)
    df= np.identity(13)
    df[0,0:9]+= dt*np.asarray([0,p1*wb3,p1*wb2,0,0,0,0,-p1*wb2*wb3,exp(-k1-k2)*wb2*wb3])
    df[1,0:9]+= dt*np.asarray([p2*wb3,0,p2*wb1,0,0,0,0,-exp(k1)*wb1*wb3,-exp(-k2)*wb1*wb3])
    df[2,0:9]+= dt*np.asarray([p3*wb2,p3*wb1,0,0,0,0,0,exp(k1+k2)*wb1*wb2,p3*wb1*wb2])
    df[3,0:7]+= dt*0.5*np.asarray([-qb1, -qb2, -qb3, 0, -wb1, -wb2, -wb3])
    df[4,0:7]+= dt*0.5*np.asarray([qb0, -qb3, qb2, wb1, 0, wb3, -wb2])
    df[5,0:7]+= dt*0.5*np.asarray([qb3, qb0, -qb1, wb2, -wb3, 0, wb1])
    df[6,0:7]+= dt*0.5*np.asarray([-qb2, qb1, qb0, wb3, wb2, -wb1, 0])

    return df

def ekf_dh_dx(x, dt):
    w1=x[0];w2=x[1];w3=x[2]
    qb0=x[3];qb1=x[4];qb2=x[5];qb3=x[6]
    qd0=x[9];qd1=x[10];qd2=x[11];qd3=x[12]

    dh = np.zeros((7,13))
    dh[0,0:3] = np.asarray([qb0**2 + qb1**2 - qb2**2 - qb3**2,2*qb1*qb2 - 2*qb0*qb3,2*qb0*qb2 + 2*qb1*qb3])
    dh[1,0:3] = np.asarray([2*qb0*qb3 + 2*qb1*qb2, qb0**2 - qb1**2 + qb2**2 - qb3**2,2*qb2*qb3 - 2*qb0*qb1])
    dh[2,0:3] = np.asarray([2*qb1*qb3 - 2*qb0*qb2,2*qb0*qb1 + 2*qb2*qb3, qb0**2 - qb1**2 - qb2**2 + qb3**2])
    dh[0,3:7] = np.asarray([2*qb0*w1 + 2*qb2*w3 - 2*qb3*w2, 2*qb1*w1 + 2*qb2*w2 + 2*qb3*w3, 2*qb0*w3 + 2*qb1*w2 - 2*qb2*w1, 2*qb1*w3 - 2*qb0*w2 - 2*qb3*w1])
    dh[1,3:7] = np.asarray([2*qb0*w2 - 2*qb1*w3 + 2*qb3*w1, 2*qb2*w1 - 2*qb1*w2 - 2*qb0*w3, 2*qb1*w1 + 2*qb2*w2 + 2*qb3*w3, 2*qb0*w1 + 2*qb2*w3 - 2*qb3*w2])
    dh[2,3:7] = np.asarray([2*qb0*w3 + 2*qb1*w2 - 2*qb2*w1, 2*qb0*w2 - 2*qb1*w3 + 2*qb3*w1, 2*qb3*w2 - 2*qb2*w3 - 2*qb0*w1, 2*qb1*w1 + 2*qb2*w2 + 2*qb3*w3])
    dh[0:3, :] *= dt
    qb = x[3:7]
    quat_one = quaternion_mul_num(qb, quat_inv(qb))
    if quat_one[0] < -0.5:
        dh[0:3, :] = -dh[0:3, :]
    
    dh[3:7, :] = np.asarray([[0, 0, 0, qd0, -qd1, -qd2, -qd3, 0, 0, qb0, -qb1, -qb2, -qb3],
                            [0, 0, 0, qd1, qd0, qd3, -qd2, 0, 0, qb1, qb0, -qb3, qb2],
                            [0, 0, 0, qd2, -qd3, qd0, qd1, 0, 0, qb2, qb3, qb0, -qb1],
                            [0, 0, 0, qd3, qd2, -qd1, qd0, 0, 0, qb3, -qb2, qb1, qb0]])

    return dh

def ekf_f(x,dt):

    dx1 = get_derivative(x)
    x_n2 = deepcopy(x)
    x_n2[0:7] = x_n2[0:7]+0.5*dt*dx1
    dx2 = get_derivative(x_n2)
    x_n3 = deepcopy(x)
    x_n3[0:7] = x_n3[0:7]+0.5*dt*dx2
    dx3 = get_derivative(x_n3)
    x_n4 = deepcopy(x)
    x_n4[0:7] = x_n4[0:7]+0.5*dt*dx3
    dx4 = get_derivative(x_n4)
    
    x_n = deepcopy(x)
    x_n[0:7] = x_n[0:7] + dt/6*(dx1+2*dx2+2*dx3+dx4)

    return x_n

def ekf_h(x_old,x,dt):
    wb_aug = np.zeros((4,))
    wb_aug[1:4] = x[0:3]
    qb = x[3:7]
    qd = x[9:13]
    quat_one = quaternion_mul_num(qb, quat_inv(qb))
    # dq = 0.5 * quaternion_mul_num(qb, wb_aug) 
    # dq = quaternion_mul_num(dq, quat_inv(qb)) * dt
    # if quat_one[0] < -0.5:
    #     dq = -dq
    h = np.zeros((7,))
    dq = quaternion_mul_num(x[3:7], quat_inv(x_old[3:7]))
    h[0:3] = dq[1:4]
    h[3:7] = quaternion_mul_num(qb, qd)
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

def ekf_predict(M,P,Q,dt):

    A = ekf_df_dx(M, dt)
    M_new = ekf_f(M, dt)
    P = A @ P @ A.T + Q

    return M_new, P

def ekf_update(M_old,M,P,y,R,dt):

    H = ekf_dh_dx(M, dt)
    MU = ekf_h(M_old, M, dt)
    S = R + H @ P @ H.T
    K = P @ H.T @ np.linalg.inv(S)
    dM = K @ (y - MU)
    M = M + dM
    y = ekf_h(M_old, M, dt)
    P = P-K @ H @ P

    return M, P, y
