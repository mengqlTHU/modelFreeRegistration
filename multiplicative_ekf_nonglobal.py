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

def mekf_df_dx(x,dt):

    wb1=x[0];wb2=x[1];wb3=x[2]
    dqb1=x[3];dqb2=x[4];dqb3=x[5]
    k1 = x[6]
    k2 = x[7]
    p1 = exp(-k1)-exp(-k1-k2)
    p2 = exp(-k2)-exp(k1)
    p3 = exp(k1+k2)-exp(k2)
    df= np.identity(11)
    df[0,0:8]+= dt*np.asarray([0,p1*wb3,p1*wb2,0,0,0,-p1*wb2*wb3,exp(-k1-k2)*wb2*wb3])
    df[1,0:8]+= dt*np.asarray([p2*wb3,0,p2*wb1,0,0,0,-exp(k1)*wb1*wb3,-exp(-k2)*wb1*wb3])
    df[2,0:8]+= dt*np.asarray([p3*wb2,p3*wb1,0,0,0,0,exp(k1+k2)*wb1*wb2,p3*wb1*wb2])
    df[3,0:6]+= dt*0.5*np.asarray([1, 0, 0, 0, wb3, -wb2])
    df[4,0:6]+= dt*0.5*np.asarray([0, 1, 0, -wb3, 0, wb1])
    df[5,0:6]+= dt*0.5*np.asarray([0, 0, 1, wb2, -wb1, 0])

    return df

def mekf_dh_dx(x, dt):
    dqb1=x[3];dqb2=x[4];dqb3=x[5]
    dqd1=x[8];dqd2=x[9];dqd3=x[10]
    qb0=x[11];qb1=x[12];qb2=x[13];qb3=x[14]
    dh = np.zeros((3,11))
    dh[0,3]=1;dh[1,4]=1;dh[2,5]=1
    dh[0,8:11] = np.asarray([qb0**2 + qb1**2 - qb2**2 - qb3**2,2*qb1*qb2 - 2*qb0*qb3,2*qb0*qb2 + 2*qb1*qb3])
    dh[1,8:11] = np.asarray([2*qb0*qb3 + 2*qb1*qb2, qb0**2 - qb1**2 + qb2**2 - qb3**2,2*qb2*qb3 - 2*qb0*qb1])
    dh[2,8:11] = np.asarray([2*qb1*qb3 - 2*qb0*qb2,2*qb0*qb1 + 2*qb2*qb3, qb0**2 - qb1**2 - qb2**2 + qb3**2])

    return dh

def mekf_f(x,dt):

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

def mekf_h(x,dt):
    dqb1=x[3];dqb2=x[4];dqb3=x[5];dqb0=sqrt(1-dqb1**2-dqb2**2-dqb3**2)
    dqd1=x[8];dqd2=x[9];dqd3=x[10];dqd0=sqrt(1-dqd1**2-dqd2*2-dqd3**2)
    qb = x[11:15]
    dq = quaternion_mul_num([dqb0,dqb1,dqb2,dqb3], qb)
    dq = quaternion_mul_num(dq, [dqd0,dqd1,dqd2,dqd3])
    dq = quaternion_mul_num(dq, quat_inv(qb))
    return dq[1:4]


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

def mekf_predict(M,P,Q,dt):

    A = mekf_df_dx(M, dt)
    M_new = mekf_f(M, dt)
    P = A @ P @ A.T + Q

    return M_new, P

def mekf_update(M,P,y,R,dt):
    dqb1=M[3];dqb2=M[4];dqb3=M[5]
    dqd1=M[8];dqd2=M[9];dqd3=M[10]
    qb = M[11:15]
    qd = M[15:19]
    # dqb_quat = np.asarray([sqrt(1-dqb1**2-dqb2**2-dqb3**2), dqb1, dqb2, dqb3])
    # dqd_quat = np.asarray([sqrt(1-dqd1**2-dqd2**2-dqd3**2), dqd1, dqd2, dqd3])
    # qb_new = quaternion_mul_num(qb, dqb_quat)
    # qd_new = quaternion_mul_num(dqd_quat, qd)

    H = mekf_dh_dx(M, dt)
    MU = mekf_h(M, dt)
    S = R + H @ P @ H.T
    K = P @ H.T @ np.linalg.inv(S)

    dM = K @ (y - MU)
    M[0:11] = M[0:11] + dM
    dqb1=M[3];dqb2=M[4];dqb3=M[5]
    dqd1=M[8];dqd2=M[9];dqd3=M[10]
    dqb_quat = np.asarray([sqrt(1-dqb1**2-dqb2**2-dqb3**2), dqb1, dqb2, dqb3])
    dqd_quat = np.asarray([sqrt(1-dqd1**2-dqd2**2-dqd3**2), dqd1, dqd2, dqd3])
    M[11:15] = quaternion_mul_num(dqb_quat, qb)
    M[15:19] = quaternion_mul_num(dqd_quat, qd)

    M[3:6] = 0
    M[8:11] = 0

    y_updated = quaternion_mul_num(M[11:15], M[15:19])
    P = P - K @ H @ P

    return M, P, y_updated

def mekf_update_without_correction(M):
    dqb1=M[3];dqb2=M[4];dqb3=M[5]
    dqd1=M[8];dqd2=M[9];dqd3=M[10]
    qb = M[11:15]
    qd = M[15:19]

    dqb_quat = np.asarray([sqrt(1-dqb1**2-dqb2**2-dqb3**2), dqb1, dqb2, dqb3])
    dqd_quat = np.asarray([sqrt(1-dqd1**2-dqd2**2-dqd3**2), dqd1, dqd2, dqd3])
    M[11:15] = quaternion_mul_num(dqb_quat, qb)
    M[15:19] = quaternion_mul_num(dqd_quat, qd)

    M[3:6] = 0
    M[8:11] = 0

    return M

