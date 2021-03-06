import open3d as o3d
import numpy as np
import copy
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from math import exp
from ekf import ekf_predict, ekf_update, quaternion_mul_num, quat_inv, ekf_h
from multiplicative_ekf import mekf_predict, mekf_update, mekf_update_without_correction
from unscented_filter import ukf_f, ukf_h, ukf_update
import rtree.index as RIndex
from bisect import bisect
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import (unscented_transform, MerweScaledSigmaPoints,
                             JulierSigmaPoints, SimplexSigmaPoints)
from filterpy.common import Q_discrete_white_noise

def visualizePC(pc):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    vis.add_geometry(pc)
    vis.run()

def generate_disturb_rotation():
    vec = np.random.normal(size=3)
    # norm_vec = vec/np.linalg.norm(vec)
    norm_vec = np.asarray([0.7,0.7,0])
    # theta = np.random.normal(scale=0.02)
    theta = 1.5
    r = Rotation.from_rotvec(abs(theta)*norm_vec)
    return r
    
def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.2, origin=[0, 0, -0.5])
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    vis.add_geometry(mesh_frame)
    vis.run()

def get_minimal_LH_norm(point_cloud):
    source_points = np.asarray(point_cloud.points)
    source_normals = np.asarray(point_cloud.normals)

    plucker_aug = np.zeros((source_points.shape[0], 6), dtype=np.float64)
    plucker_aug[:,0:3] = source_normals

    for j in range(source_points.shape[0]):
        plucker_aug[j, 3:6] = np.cross(source_points[j,:], source_normals[j,:])    
    u, s, vh = np.linalg.svd(plucker_aug)
    h_optimal = vh.T[:,-1]
    min_LH_norm = np.linalg.norm(plucker_aug @ h_optimal)

    return min_LH_norm

def loop_closure_correction(source, target, source_R):
    current_transformation = np.identity(4)
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, 0.05, current_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-12,
                                    relative_rmse=1e-12,
                                    max_iteration=50))
    dR_estimate = Rotation.from_matrix(result_icp.transformation[:3,:3])
    R_correct = dR_estimate * source_R 
    return R_correct

def continous_quat(pose_data):
    for i in range(1,pose_data.shape[0]):
        if np.linalg.norm(pose_data[i,6:10] - pose_data[i-1,6:10]) > 0.5:
            pose_data[i,6:10] = -pose_data[i,6:10]
    return pose_data

def to_scalar_last(q):
    q_new = np.zeros((4,))
    q_new[3] = q[0]
    q_new[0:3] = q[1:4]
    return q_new

def to_scalar_first(q):
    q_new = np.zeros((4,))
    q_new[0] = q[3]
    q_new[1:4] = q[0:3]
    return q_new 

def normalize_state(q_in):
    q_in[3:7] = q_in[3:7]/np.linalg.norm(q_in[3:7])
    q_in[9:13] = q_in[9:13]/np.linalg.norm(q_in[9:13])
    return q_in

def mekf_normalize_state(q_in):
    q_in[11:15] = q_in[11:15]/np.linalg.norm(q_in[11:15])
    q_in[15:19] = q_in[15:19]/np.linalg.norm(q_in[15:19])
    return q_in  

def get_quat_angle(q_in):
    r = Rotation.from_quat(to_scalar_last(q_in))
    drotvec_error = r.as_rotvec()
    return np.linalg.norm(drotvec_error)

def register_and_filter_once(x, filt_times, has_measurement):
    start_frame = 12
    end_frame = 12

    filepath = './20200112_notate/'
    pose_data = np.genfromtxt(f'{filepath}point_cloud_pose.txt', delimiter=',')
    timestamp_data = np.genfromtxt(f'{filepath}timestamp.txt', delimiter=',')

    jump_step = 10
    downsample_transformation = np.identity(4)
    pose_data = pose_data[start_frame:-end_frame:1,:]
    timestamp_data = timestamp_data[start_frame:-end_frame:1,:]

    pose_data = continous_quat(pose_data)

    quat0_inv = quat_inv(pose_data[0,6:10])
    for i in range(pose_data.shape[0]):
        pose_data[i, 6:10] = quaternion_mul_num(pose_data[i, 6:10], quat0_inv)

    # In reverse Order
    if filt_times%2==0:
        pose_data = pose_data[::-1,:]
        timestamp_data = timestamp_data[::-1,:]

    source_quat_zero_raw = pose_data[0,6:10]
    source_quat_zero = np.zeros((4,))
    source_quat_zero[3] = source_quat_zero_raw[0]
    source_quat_zero[0:3] = source_quat_zero_raw[1:4]
    sourceR_zero = Rotation.from_quat(source_quat_zero)

    loop_start_R = sourceR_zero
    loop_start_point_cloud = None
    need_loop_check = False
    need_loop_check_state = 0
    loop_detected = False
    tmp_pc_stored = False
    rtreep = RIndex.Property()
    rtreep.dimension = 3
    rtree = RIndex.Index(properties=rtreep)
    loop_buffer_index = 0

    now_quat = source_quat_zero
    register_log =  np.zeros((pose_data.shape[0],14), dtype=np.float64)
    register_log[0, 0:4] = to_scalar_first(now_quat)
    measurement_log = np.zeros((pose_data.shape[0],7), dtype=np.float64)
    measurement_log[0, 3:7] = source_quat_zero_raw

    nowR = Rotation.from_quat(now_quat)
    nowR_jump = copy.deepcopy(nowR)

    ## EKF SETTING
    R = 1e-5*np.diag(np.asarray([1,1,1]))
    P = 1e-4*np.diag(np.asarray([0.0001,0.0001,1,1,1,1,10,10,1,1,1]))
    Q = 1e-8*np.diag(np.asarray([0.0001,0.0001,1,1,1,1,10,10,1,1,1]))

    x_log = np.zeros((pose_data.shape[0],19))
    x_log[0,:] = x

    source_s_root = 1
    jumped_i = 0

    for i in range(pose_data.shape[0]-1):
        print("1. Load two point clouds and show initial pose")
        source = o3d.io.read_point_cloud(f"{filepath}Cropped_Frame{int(timestamp_data[i,0]-1)}.pcd")
        target = o3d.io.read_point_cloud(f"{filepath}Cropped_Frame{int(timestamp_data[i+1,0]-1)}.pcd")

        source,_ = source.remove_statistical_outlier(100, 2.0)
        target,_ = target.remove_statistical_outlier(100, 2.0)

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

        dR_real = targetR * sourceR.inv()
        drotvec_real = dR_real.as_rotvec()
        
        dtime = abs(timestamp_data[i+1, 1] - timestamp_data[i, 1])
        x_predict, P = mekf_predict(x, P, Q, dtime)

        o3d.geometry.PointCloud.estimate_normals(target,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01,max_nn=30))

        # point to point ICP
        current_transformation = np.identity(4)

        radius = 0.01
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=15))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=15))
        
        if i == 0:
            source_points = np.asarray(source_down.points)
            source_normals = np.asarray(source_down.normals)

            plucker_aug = np.zeros((source_points.shape[0], 6), dtype=np.float64)
            plucker_aug[:,0:3] = source_normals

            for j in range(source_points.shape[0]):
                plucker_aug[j, 3:6] = np.cross(source_points[j,:], source_normals[j,:])
            
            u, s, vh = np.linalg.svd(plucker_aug)
            source_s_root = s[5]
            loop_start_point_cloud = copy.deepcopy(source)

        target_points = np.asarray(target_down.points)
        target_normals = np.asarray(target_down.normals)

        plucker_aug = np.zeros((target_points.shape[0], 6), dtype=np.float64)
        plucker_aug[:,0:3] = target_normals

        for j in range(target_points.shape[0]):
            plucker_aug[j, 3:6] = np.cross(target_points[j,:], target_normals[j,:])
        
        u, s, vh = np.linalg.svd(plucker_aug)
        single_value_min = min(source_s_root, s[5])
        source_s_root = s[5]

        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, 0.03, current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        dR_estimate = Rotation.from_matrix(result_icp.transformation[:3,:3])

        dR_error =  dR_real * dR_estimate.inv()
        drotvec_error = dR_error.as_rotvec()
        print(np.linalg.norm(drotvec_error))
        if np.linalg.norm(drotvec_error)>0.1:
            aaa = 5
        dR_estimate_angle = np.linalg.norm(dR_estimate.as_rotvec())

        downsample_transformation = result_icp.transformation @ downsample_transformation

        if single_value_min > 0.3 and dR_estimate_angle<0.3:
            dq_estimate = dR_estimate.as_quat()
            if dq_estimate[3]>0.5:
                dq_estimate = dq_estimate
            else:
                dq_estimate = -dq_estimate
            z_measure = np.zeros((7,))
            z_measure[0:3] = dq_estimate[0:3]

            nowR = dR_estimate * Rotation.from_quat(now_quat)

            #Do Loop Closure Check
            nowR_from_loop_start = nowR * loop_start_R.inv()
            nowR_from_loop_start_angle = np.linalg.norm(nowR_from_loop_start.as_rotvec())
            if not need_loop_check and nowR_from_loop_start_angle > 1:
                need_loop_check = True
            
            print(nowR_from_loop_start_angle)
            if need_loop_check and nowR_from_loop_start_angle < 0.5 and single_value_min > 0.5:
                nowR = loop_closure_correction(loop_start_point_cloud, target, loop_start_R)
                # loop_start_R = nowR
                # loop_start_point_cloud = copy.deepcopy(target)
                need_loop_check = False
                loop_detected = True

            if i-jumped_i>=jump_step:
                if single_value_min > 0.45:
                    if not loop_detected:
                        source_jump = o3d.io.read_point_cloud(f"{filepath}Cropped_Frame{int(timestamp_data[jumped_i,0]-1)}.pcd")
                        source_quat_jump = pose_data[jumped_i,6:10]
                        sourceR_jump = Rotation.from_quat(to_scalar_last(source_quat_jump))
                        dR_jump_real = targetR * sourceR_jump.inv()

                        source_jump,_ = source_jump.remove_statistical_outlier(100, 2.0)

                        source_jump.estimate_normals(
                        o3d.geometry.KDTreeSearchParamHybrid(radius=0.002, max_nn=15))

                        result_icp_temp = o3d.pipelines.registration.registration_icp(
                            source_jump, target, 0.05, downsample_transformation)

                        result_icp_jump = o3d.pipelines.registration.registration_icp(
                            source_jump, target, 0.005, result_icp_temp.transformation,
                            o3d.pipelines.registration.TransformationEstimationPointToPlane()
                            ,o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-12,
                                                                relative_rmse=1e-12,
                                                                max_iteration=50))

                        dR_jump_estimate = Rotation.from_matrix(result_icp_jump.transformation[:3,:3])
                        dR_jump_error =  dR_jump_real * dR_jump_estimate.inv()
                        drotvec_jump_error = dR_jump_error.as_rotvec()
                        if np.linalg.norm(drotvec_jump_error)>0.1:
                            kkk = 8
                        nowR = dR_jump_estimate * nowR_jump
                        jumped_i = i + 1
                        nowR_jump = Rotation.from_matrix(nowR.as_matrix())
                        downsample_transformation = np.identity(4)
                    else:
                        loop_detected = False
                

            if np.linalg.norm(now_quat - nowR.as_quat())>0.5:
                tmp_quat = -nowR.as_quat()
            else:
                tmp_quat = nowR.as_quat()
            z_measure[3:7] = to_scalar_first(tmp_quat)
            
            x_correct, P, z_correct = mekf_update(x_predict, P, z_measure[3:7], R, dtime)
  
            if np.linalg.norm(now_quat - nowR.as_quat())>0.5:
                now_quat = -nowR.as_quat()
            else:
                now_quat = nowR.as_quat()

            x_correct = mekf_normalize_state(x_correct)
            x_log[i+1,:] = x_correct
            x = x_correct

            dR_error_cumulative = targetR  * nowR.inv()
            # dR_error_cumulative = nowR.inv() * targetR
            drotvec_error_cumulative = dR_error_cumulative.as_rotvec()
            register_log[i+1,0:4] = to_scalar_first(now_quat)
            register_log[i+1,4] = np.linalg.norm(drotvec_error)
            register_log[i+1,5] = np.linalg.norm(drotvec_error_cumulative)
            register_log[i+1,6] = single_value_min
            register_log[i+1,7] = nowR_from_loop_start_angle
            register_log[i+1,8:11] = dR_error.as_euler("zyx")
            register_log[i+1,11:14] = dR_error_cumulative.as_euler("zyx")

            print(register_log[i+1,5])
        else:

            if not tmp_pc_stored:
                tmp_pc_stored = True
                tmp_pc = copy.deepcopy(source)

            x_predict = mekf_update_without_correction(x_predict)
            x_predict = mekf_normalize_state(x_predict)
            x_log[i+1,:] = x_predict

            nowR = Rotation.from_quat(to_scalar_last(quaternion_mul_num(x_predict[11:15], x_predict[15:19])))
            if np.linalg.norm(now_quat - nowR.as_quat())>0.5:
                now_quat = -nowR.as_quat()
            else:
                now_quat = nowR.as_quat()

            x = x_predict

            dR_error_cumulative = targetR * nowR.inv()
            # dR_error_cumulative = nowR.inv() * targetR
            drotvec_error_cumulative = dR_error_cumulative.as_rotvec()
            register_log[i+1,0:4] = to_scalar_first(now_quat)
            register_log[i+1,4] = np.linalg.norm(drotvec_error)
            register_log[i+1,5] = np.linalg.norm(drotvec_error_cumulative)
            register_log[i+1,6] = single_value_min
            register_log[i+1,7] = nowR_from_loop_start_angle
            register_log[i+1,8:11] = dR_error.as_euler("zyx")
            register_log[i+1,11:14] = dR_error_cumulative.as_euler("zyx")

            print(register_log[i+1,5])

        measurement_log[i+1, 0:3] = dq_estimate[0:3]
        measurement_log[i+1, 3:7] = to_scalar_first(now_quat)
        # draw_registration_result_original_color(source, target, np.identity(4))

    # np.savetxt(f"{filepath}ekf_log_all_{filt_times}.csv", x_log, delimiter=',')
    np.savetxt(f"{filepath}log_all_correction_{filt_times}.csv", register_log, delimiter=',')

    # np.savetxt(f"{filepath}registration_measurement.csv", measurement_log, delimiter=',')

    plt.figure()
    plt.plot(register_log[:,0:4])
    plt.plot(pose_data[:, 6:10])
    plt.show()

    return x, measurement_log

def mekf_filter_without_register(x, filt_times, filepath, measurement_raw, timestamp_data):

    measurement = copy.deepcopy(measurement_raw)
    # In reverse Order
    if filt_times%2==0:
        measurement = measurement[::-1, :]
        timestamp_data = timestamp_data[::-1, :]
        for i in range(1, measurement.shape[0]):
            measurement[measurement.shape[0] - i, 0:3] = -measurement[measurement.shape[0] - i - 1, 0:3]
            # measurement[i, 0:3] = -measurement[i, 0:3]

    R = 1e-5*np.diag(np.asarray([1,1,1]))
    if filt_times == 1:
        P = 1e-4*np.diag(np.asarray([0.0001,0.0001,1,1,1,1,10,10,1,1,1]))
        Q = 1e-8*np.diag(np.asarray([0.0001,0.0001,1,1,1,1,10,10,1,1,1]))
    else:
        P = 1e-4*np.diag(np.asarray([0.001,0.001,0.001,1,1,1,10,10,1,1,1]))
        Q = 1e-8*np.diag(np.asarray([0.001,0.001,0.001,1,1,1,10,10,1,1,1]))
        shrink_list = [3]
        shrink_index = bisect(shrink_list, filt_times)
        P[6:8,:] = P[6:8,:]/(10**shrink_index)
        Q[6:8,:] = Q[6:8,:]/(10**shrink_index)
        # P[8:11,:] = P[8:11,:]/(10**shrink_index)
        # Q[8:11,:] = Q[8:11,:]/(10**shrink_index)
        # P[3:7] = P[3:7]/(10**shrink_index)
        # Q[3:7] = Q[3:7]/(10**shrink_index)

    x_log = np.zeros((measurement.shape[0],19))
    x_log[0,:] = x

    for i in range(measurement.shape[0]-1):
        
        dtime = abs(timestamp_data[i+1, 1] - timestamp_data[i, 1])
        x_predict, P = mekf_predict(x, P, Q, dtime)


        # z_measure[3:7] = target_quat_raw
        z_measure = measurement[i + 1, 3:7]
        x_correct, P, z_correct = mekf_update(x_predict, P, z_measure, R, dtime)
        
        x_correct = mekf_normalize_state(x_correct)

        x_log[i+1,:] = x_correct
        x = x_correct

    np.savetxt(f"{filepath}mekf_log_all_{filt_times}.csv", x_log, delimiter=',')
    return x

if __name__ == '__main__':
    # x0 = np.zeros((13,))
    # x0[9] = 1
    x0 = np.zeros((19,))
    x0[11] = 1
    x0[15] = 1
    x_end, _ = register_and_filter_once(x0, 1, False)
    x_end[0:3] = -x_end[0:3]
    x_end, _ = register_and_filter_once(x_end, 2, True)
    x_end[0:3] = -x_end[0:3]
    x_end, _ = register_and_filter_once(x_end, 3, True)
    x_end[0:3] = -x_end[0:3]
    x_end, _ = register_and_filter_once(x_end, 4, True)
    x_end[0:3] = -x_end[0:3]
    x_end, _ = register_and_filter_once(x_end, 5, True)
    x_end[0:3] = -x_end[0:3]
    x_end, _ = register_and_filter_once(x_end, 6, True)
    x_end[0:3] = -x_end[0:3]
    x_end, _ = register_and_filter_once(x_end, 7, True)

    I_real = np.asarray([20, 50, 50])
    I_real = I_real/np.linalg.norm(I_real)
    k1, k2 = x_end[6], x_end[7]
    I_estimate = np.sort(np.asarray([exp(k1), 1, exp(-k2)]))
    I_estimate = I_estimate/np.linalg.norm(I_estimate)

    print(I_real)
    print(I_estimate)

