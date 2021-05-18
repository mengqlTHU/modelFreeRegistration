import numpy as np
import copy
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from math import exp
# from ekf import ekf_predict, ekf_update, quaternion_mul_num, quat_inv
from multiplicative_ekf_nonglobal import mekf_predict, mekf_update, mekf_update_without_correction, quat_inv,quaternion_mul_num
from unscented_filter import ukf_f, ukf_h, ukf_update
from bisect import bisect
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

def get_w_in_body_frame(q1, q2, dt):
    q_dot = (q2-q1)/dt
    w1 = quaternion_mul_num(quat_inv(q1), 2*q_dot)
    w2 = quaternion_mul_num(quat_inv(q2), 2*q_dot)
    w = (w1+w2)/2
    return w[1:4]

def to_scalar_first(q):
    q_new = np.zeros((4,))
    q_new[0] = q[3]
    q_new[1:4] = q[0:3]
    return q_new 

def to_scalar_last(q):
    q_new = np.zeros((4,))
    q_new[3] = q[0]
    q_new[0:3] = q[1:4]
    return q_new 

def continous_quat(pose_data):
    for i in range(1,pose_data.shape[0]):
        if np.linalg.norm(pose_data[i,6:10] - pose_data[i-1,6:10]) > 0.5:
            pose_data[i,6:10] = -pose_data[i,6:10]
    return pose_data

def normalize_state(q_in):
    q_in[3:7] = q_in[3:7]/np.linalg.norm(q_in[3:7])
    q_in[9:13] = q_in[9:13]/np.linalg.norm(q_in[9:13])
    return q_in

def mekf_normalize_state(q_in):
    q_in[11:15] = q_in[11:15]/np.linalg.norm(q_in[11:15])
    q_in[15:19] = q_in[15:19]/np.linalg.norm(q_in[15:19])
    return q_in  

def register_and_filter_once(x, filt_times):
    start_frame = 48
    end_frame = 12
    skip = 1

    filepath = "./20210312_3axis/"
    pose_data = np.genfromtxt(f'{filepath}point_cloud_pose.txt', delimiter=',')
    timestamp_data = np.genfromtxt(f'{filepath}timestamp.txt', delimiter=',')

    pose_data = pose_data[start_frame:-end_frame:skip,:]
    measurement_data = np.genfromtxt(f'{filepath}registration_measurement.csv', delimiter=',')
    measurement_data = measurement_data[::skip,:]
    pose_data[:, 6:10] = measurement_data[::skip,3:7]
    timestamp_data = timestamp_data[start_frame:-end_frame:skip,:]
    pose_data = continous_quat(pose_data)

    # filepath = "./test_pointcloud/"
    # pose_data = np.genfromtxt(f'{filepath}blensor_data_csv.txt', delimiter=',')
    # pose_data = np.concatenate((np.zeros((pose_data.shape[0],6)), pose_data), axis=1)
    # arange_data = np.arange(pose_data.shape[0]).reshape((pose_data.shape[0],1))
    # timestamp_data = np.concatenate((arange_data, arange_data*0.05), axis=1)
    # pose_data = pose_data[::skip,:]
    # timestamp_data = timestamp_data[::skip,:]

    quat0_inv = quat_inv(pose_data[0,6:10])
    for i in range(pose_data.shape[0]):
        pose_data[i, 6:10] = quaternion_mul_num(pose_data[i, 6:10], quat0_inv)

    # In reverse Order
    if filt_times%2==0:
        pose_data = pose_data[::-1,:]
        timestamp_data = timestamp_data[::-1,:]
        measurement_data[:,0:3] = -measurement_data[::-1,0:3]

    source_quat_zero_raw = pose_data[0,6:10]
        
    R = 1e-6*np.diag(np.asarray([1,1,1]))
    P = 1e-4*np.diag(np.asarray([0.001,0.001,0.001,1,1,1,10,10,1,1,1]))
    Q = 1e-8*np.diag(np.asarray([0.001,0.001,0.001,1,1,1,10,10,1,1,1]))
    # shrink_list = [3, 5, 7]
    # shrink_index = bisect(shrink_list, filt_times)
    # P[7:9,:] = P[7:9,:]/(10**shrink_index)
    # Q[7:9,:] = Q[7:9,:]/(10**shrink_index)

    # P[0:6]*=skip**2
    # Q[0:6]*=skip**2
    P[0:3]*=1000
    Q[0:3]*=1000
    x_log = np.zeros((pose_data.shape[0],19))
    x_log[0,:] = x

    #UKF setting
    points = MerweScaledSigmaPoints(11, alpha=.1, beta=2., kappa=-1)
    ukf = UKF(dim_x=11, dim_z=3, dt=0.1, fx=ukf_f, hx=ukf_h, points=points)
    ukf.x = x[0:11] # initial state
    ukf.P = P
    ukf.Q = Q
    ukf.R = R
    x_global = x[11:19]

    # w_body = np.zeros((pose_data.shape[0]-1, 3))
    # for i in range(w_body.shape[0]):
    #     w_body[i, :] = get_w_in_body_frame(pose_data[i,6:10], pose_data[i+1,6:10], timestamp_data[i+1, 1] - timestamp_data[i, 1])

    source_s_root = 1
    now_quat = source_quat_zero_raw
    register_log = np.zeros((pose_data.shape[0],4))
    dq_real_log = np.zeros((pose_data.shape[0]-1,4))
    dq_correct_log = np.zeros((pose_data.shape[0]-1,4))
    register_log[0,:] = now_quat
    loop_start_R = Rotation.from_quat(to_scalar_last(now_quat))
    loop_angle_log = np.zeros((pose_data.shape[0],))
    for i in range(pose_data.shape[0]-1):

        source_quat_raw = pose_data[i,6:10]
        target_quat_raw = pose_data[i+1,6:10]
        source_quat = np.zeros((4,))
        target_quat = np.zeros((4,))
        source_quat[3] = source_quat_raw[0]
        source_quat[0:3] = source_quat_raw[1:4]
        target_quat[3] = target_quat_raw[0]
        target_quat[0:3] = target_quat_raw[1:4]
        sourceR = Rotation.from_quat(source_quat)
        targetR = Rotation.from_quat(target_quat)

        nowR_from_loop_start = targetR * loop_start_R.inv()
        nowR_from_loop_start_angle = np.linalg.norm(nowR_from_loop_start.as_rotvec())
        loop_angle_log[i] = nowR_from_loop_start_angle

        dR_real = targetR * sourceR.inv()
        dq_real = to_scalar_first(dR_real.as_quat())
        dq_real_log[i,:] = dq_real
        
        dtime = abs(timestamp_data[i+1, 1] - timestamp_data[i, 1])
        # x_predict, P = ekf_predict(x, P, Q, dtime)
        x_predict, P = mekf_predict(x, P, Q, dtime)
        # ukf.predict(dt=dtime)

        # point to plane ICP
        dq_estimate = dR_real.as_quat()
        if dq_estimate[3]>0.5:
            dq_estimate = dq_estimate
        else:
            dq_estimate = -dq_estimate
        z_measure = np.zeros((7,))
        z_measure[0:3] = dq_estimate[0:3]
        # z_measure[3:7] = quaternion_mul_num(to_scalar_first(dq_estimate), now_quat)
        z_measure[3:7] = target_quat_raw

        # x_correct, P, z_correct = ekf_update(x, x_predict, P, z_measure, R, dtime)
        x_correct, P, z_correct = mekf_update(x_predict, P, measurement_data[i+1,0:3], R, dtime)
        # ukf, x_global, _ = ukf_update(ukf, z_measure[3:7], x_global)

        # dq_correct = np.asarray([z_correct[0],z_correct[1],z_correct[2],1])
        # dR_correct = Rotation.from_quat(dq_correct/np.linalg.norm(dq_correct))

        x_correct = mekf_normalize_state(x_correct)
        x_log[i+1,:] = x_correct
        x = x_correct

        # x_global[0:4] = x_global[0:4]/np.linalg.norm(x_global[0:4])
        # x_global[4:8] = x_global[4:8]/np.linalg.norm(x_global[4:8])
        # x_log[i+1,0:11] = ukf.x
        # x_log[i+1,11:19] = x_global

        # now_quat = quaternion_mul_num(to_scalar_first(dq_correct), now_quat)
        # now_quat = z_correct[3:7]
        # register_log[i+1, :] = now_quat
        # dq_correct_log[i,:] = to_scalar_first(dq_correct)

    np.savetxt(f"{filepath}mekf_log_ground_truth_{filt_times}.csv", x_log, delimiter=',')

    # x[0:11] = ukf.x
    # x[11:19] = x_global
    return x

if __name__=='__main__':
    x0 = np.zeros((19,))
    x0[11] = 1
    x0[15] = 1
    x_end = x0
    for i in range(9):
        x_end = register_and_filter_once(x_end, i+1)
        x_end[0:3] = -x_end[0:3]
    x_end[0:3] = -x_end[0:3] 

    I_real = np.asarray([20, 35, 50])
    I_real = I_real/np.linalg.norm(I_real)
    k1, k2 = x_end[6], x_end[7]
    I_estimate = np.sort(np.asarray([exp(k1), 1, exp(-k2)]))
    I_estimate = I_estimate/np.linalg.norm(I_estimate)

    print(I_real)
    print(I_estimate)