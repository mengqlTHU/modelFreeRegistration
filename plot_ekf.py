import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.patches import Rectangle
from continous_register import to_scalar_first, to_scalar_last, continous_quat
from ekf import quaternion_mul_num, quat_inv
import math
from scipy.spatial.transform import Rotation

def get_w_in_body_frame(q1, q2, dt):
  q_dot = (q2-q1)/dt
  w1 = quaternion_mul_num(quat_inv(q1),2*q_dot)
  w2 = quaternion_mul_num(quat_inv(q2),2*q_dot)
  w = (w1+w2)/2
  return w[1:]

filepath = './20200112_notate/'
qd_origin = np.asarray([0.621143100710754,0.392048569170185,-0.338417655549184,-0.588177403734311])
qd_true = np.asarray([0.849038414608915,0.510562868421641,0.0862028270363913,0.103448830435052])
qd_rotate = np.asarray([0.6375,0,0,-0.7702])
Rd_rotate = Rotation.from_quat(to_scalar_last(qd_rotate)).as_matrix()
start_frame = 12
end_frame = 12
I_real = np.asarray([50, 50, 20])
I_real = I_real/np.linalg.norm(I_real)
matlab_interval = 0.05

# filepath = './20200205_3axis/'
# qd_origin = np.asarray([0.0569346289156478,-0.0831563740366123,-0.534127353694238,-0.839375622425926])
# qd_true = np.asarray([0.839375622425926,0.534127353694238,-0.0831563740366123,0.0569346289156478])
# qd_rotate = np.asarray([0,0,0,-1])
# Rd_rotate = Rotation.from_quat(to_scalar_last(qd_rotate)).as_matrix()
# start_frame = 33
# end_frame = 12
# I_real = np.asarray([50, 35, 20])
# I_real = I_real/np.linalg.norm(I_real)
# matlab_interval = 0.05

# filepath = './20210312_3axis/'
# qd_origin = np.asarray([0.14,-0.03,-0.53,-0.83])
# qd_true = np.asarray([0.839407665963232,0.534152422208151,-0.0759259370839453,0.0656598221841145])
# qd_rotate = np.asarray([0,0,0,-1])
# Rd_rotate = Rotation.from_quat(to_scalar_last(qd_rotate)).as_matrix()
# start_frame = 48
# end_frame = 12
# I_real = np.asarray([50, 35, 20])
# I_real = I_real/np.linalg.norm(I_real)
# matlab_interval = 0.05

ekf_log = np.genfromtxt(f'{filepath}ekf_log_all_5.csv',delimiter=',')
# ekf_log = np.genfromtxt(f'{filepath}ekf_log_ground_truth_7.csv',delimiter=',')
pose_data = np.genfromtxt(f'{filepath}point_cloud_pose.txt', delimiter=',')
timestamp_data = np.genfromtxt(f'{filepath}timestamp.txt', delimiter=',')
wb_real_data = np.genfromtxt(f'{filepath}omega_body.csv', delimiter=',')
measure_data = np.genfromtxt(f'{filepath}registration_measurement.csv', delimiter=',')

pose_data = pose_data[start_frame:-end_frame:1,:]
timestamp_data = timestamp_data[start_frame:-end_frame:1,:]

pose_data = continous_quat(pose_data)

plt.rcParams.update({"font.size" : 40})
plt.rcParams.update({"font.sans-serif" : "Arial"})
plt.rcParams.update({"font.family" : "sans-serif"})

# quat0_inv = quaternion_mul_num([0.998009114291437,0.00513295678935018,0.00686830899720483,0.0624842930450399], quat_inv(pose_data[0,6:10]))
# for i in range(pose_data.shape[0]):
#     # pose_data[i, 6:10] = quaternion_mul_num(pose_data[i, 6:10], quat0_inv)
#     pose_data[i, 6:10] = quaternion_mul_num(quat0_inv, pose_data[i, 6:10])

# plt.figure()
# plt.plot(pose_data[:, 6:10])
# plt.show()

quat0_inv = quat_inv(pose_data[0,6:10])
qb_real = np.zeros((pose_data.shape[0], 4))
wb_real = np.zeros((pose_data.shape[0], 3))
wt_real = np.zeros((pose_data.shape[0], 3))
wt_estimate = np.zeros((pose_data.shape[0], 3))
wt_measure = np.zeros((pose_data.shape[0], 3))
I_estimate = np.zeros((pose_data.shape[0], 3))
qt_estimate = np.zeros((pose_data.shape[0], 4))
qt_real = np.zeros((pose_data.shape[0], 4))

for i in range(pose_data.shape[0]):
    qt_estimate[i, :] = quaternion_mul_num(ekf_log[i, 3:7], ekf_log[i, 9:13])
    qt_estimate[i,:] = qt_estimate[i,:]/np.linalg.norm(qt_estimate[i,:])
    qt_real[i, :] = quaternion_mul_num(pose_data[i, 6:10], qd_origin)
    qt_real[i,:] = qt_real[i,:]/np.linalg.norm(qt_real[i,:])

for i in range(pose_data.shape[0]):
    qb_real[i, :] = quaternion_mul_num(pose_data[i, 6:10], qd_rotate)

for i in range(1, pose_data.shape[0]-1):
    time_now = timestamp_data[i, 1].item()
    time_index = math.ceil(time_now/matlab_interval)
    wb_real[i,:] = (Rd_rotate.T @ wb_real_data[time_index, :].T).T
    R_qb_real = Rotation.from_quat(to_scalar_last(qb_real[i,:])).as_matrix()
    wt_real[i, :] = (R_qb_real @ wb_real[i,:].T).T

for i in range(1, pose_data.shape[0]-1):
    R_qb_estimate = Rotation.from_quat(to_scalar_last(ekf_log[i,3:7])).as_matrix()
    wt_estimate[i, :] = (R_qb_estimate @ ekf_log[i,0:3].T).T
    wt_measure[i, :] = 2*measure_data[i, 0:3]/(timestamp_data[i, 1] - timestamp_data[i-1, 1])

# plt.figure()
# plt.plot(wt_estimate)
# plt.plot(wt_measure)
# plt.show()


# for i in range(1, pose_data.shape[0]-1):
#     wb_real[i, :] = get_w_in_body_frame(ekf_log[i-1,3:7], ekf_log[i+1,3:7], timestamp_data[i+1, 1]-timestamp_data[i-1, 1])

for i in range(pose_data.shape[0]):
    k1, k2 = ekf_log[i, 7], ekf_log[i, 8]
    I_estimate[i, :] = np.asarray([math.exp(k1), 1, math.exp(-k2)])
    I_estimate[i, :] = I_estimate[i, :]/np.linalg.norm(I_estimate[i,:])

print(I_real)
print(I_estimate[-1,:])
print(np.divide(I_estimate[-1,:]-I_real,I_real))
print(qd_true)
print(ekf_log[-1,9:13])
print(quaternion_mul_num(qd_true, quat_inv(ekf_log[-1,9:13])))

fig=plt.figure(1, figsize=(16,9))
fig.subplots_adjust(bottom=0.15)
fig.subplots_adjust(top=0.99)
fig.subplots_adjust(right=0.99)
color_list = ["tab:blue", "tab:orange", "tab:green"]
var_name = ["$\omega_{px}$", "$\omega_{py}$", "$\omega_{pz}$"]
for i in range(3):
    plt.plot(timestamp_data[1:-1,1], wb_real[1:-1, i], '--', linewidth=3, color=color_list[i])
    plt.plot(timestamp_data[1:-1,1], ekf_log[1:-1, i], linewidth=3, color=color_list[i], label=var_name[i])
plt.xlabel('Time(s)')
plt.ylabel('$\mathbf{\omega_{p}} (rad/s)$')
plt.xticks(np.arange(0,121,step=20))
plt.grid()
plt.legend()
plt.show()

fig=plt.figure(2, figsize=(16,9))
fig.subplots_adjust(bottom=0.15)
fig.subplots_adjust(top=0.99)
fig.subplots_adjust(right=0.99)
color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for i in range(4):
    plt.plot(timestamp_data[1:-1,1], qb_real[1:-1, i], '--', linewidth=3, color=color_list[i])
    plt.plot(timestamp_data[1:-1,1], ekf_log[1:-1, 3+i], linewidth=3, color=color_list[i], label=f"$q_{{b}}{{}}_{i}$")
plt.xlabel('Time(s)')
plt.ylabel('$\mathbf{q_{b}}$')
plt.xticks(np.arange(0,121,step=20))
plt.grid()
plt.legend()
plt.show()

# plt.figure(3)
# plt.plot(timestamp_data[1:-1,1], (np.ones((pose_data.shape[0], 3)) @ np.diag(I_real))[1:-1, :], '--')
# plt.plot(timestamp_data[1:-1,1], I_estimate[1:-1, :])
# plt.show()

# plt.figure(4)
# plt.plot(timestamp_data[1:-1,1], (np.ones((pose_data.shape[0], 4)) @ np.diag(qd_true))[1:-1, :], '--')
# plt.plot(timestamp_data[1:-1,1], ekf_log[1:-1, 9:13])
# plt.show()

# plt.figure(4)
# # plt.plot(timestamp_data[1:-1,1], qt_real[1:-1, :])
# plt.plot(timestamp_data[1:-1,1], measure_data[1:-1, 4:7])
# plt.plot(timestamp_data[1:-1,1], qt_real[1:-1, :])
# plt.show()

ekf_log = None
for i in range(5):
    if ekf_log is None:
        ekf_log = np.genfromtxt(f'{filepath}ekf_log_all_{i+1}.csv',delimiter=',')
    else:
        ekf_log = np.concatenate((ekf_log, np.genfromtxt(f'{filepath}ekf_log_all_{i+1}.csv',delimiter=',')))

I_estimate = np.zeros((ekf_log.shape[0], 3))

for i in range(ekf_log.shape[0]):
    k1, k2 = ekf_log[i, 7], ekf_log[i, 8]
    I_estimate[i, :] = np.asarray([math.exp(k1), 1, math.exp(-k2)])
    I_estimate[i, :] = I_estimate[i, :]/np.linalg.norm(I_estimate[i,:])

fig, ax = plt.subplots(figsize=(16,9))
fig.subplots_adjust(bottom=0.15)
fig.subplots_adjust(top=0.99)
fig.subplots_adjust(right=0.99)
color_list = ["tab:blue", "tab:orange", "tab:green"]
for i in range(3):
    ax.plot((np.ones((ekf_log.shape[0],1)) * I_real[i])[1:-1,:], '--', linewidth=3, color=color_list[i])
    ax.plot(I_estimate[1:-1, i], linewidth=3, label=f"$I_{i+1}$", color=color_list[i])
plt.xlabel('Iteration')
plt.ylabel('$\mathbf{\\bar{I}} $')
for i in range(5):
    width = pose_data.shape[0]
    left = width * i
    if i%2==1:
        ax.add_patch(Rectangle((left, 0), width, 1, color="lightblue"))
    ax.text(left+width/2, 0.1, f"{i+1}")
plt.yticks(np.arange(0,1.01,step=0.2))
plt.legend(loc="upper right", prop={"size":40})
plt.show()


fig, ax = plt.subplots(figsize=(16,9))
fig.subplots_adjust(bottom=0.15)
fig.subplots_adjust(top=0.99)
fig.subplots_adjust(right=0.99)
color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for i in range(4):
    ax.plot((np.ones((ekf_log.shape[0],1)) * qd_true[i])[1:-1,:], '--', linewidth=3, color=color_list[i])
    ax.plot(ekf_log[1:-1, 9+i], linewidth=3, label=f"$q_{{d}}{{}}_{i}$", color=color_list[i])
plt.xlabel('Iteration')
plt.ylabel('$\mathbf{q_d} $')
for i in range(5):
    width = pose_data.shape[0]
    left = width * i
    if i%2==1:
        ax.add_patch(Rectangle((left, -0.3), width, 1.3, color="lightblue"))
    ax.text(left+width/2, -0.2, f"{i+1}")

plt.yticks(np.arange(-0.3,1.01,step=0.2))
plt.legend(loc="upper right", prop={"size":40})
plt.show()




