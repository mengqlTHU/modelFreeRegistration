import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

# filepath = './20210312_3axis/'
# register_log = np.genfromtxt(f'{filepath}log_all_1.csv',delimiter=',')
# register_log_no_correction = np.genfromtxt(f'{filepath}log_no_correction_1.csv',delimiter=',')
# register_log_only_loop = np.genfromtxt(f'{filepath}log_only_loop_correction_1.csv',delimiter=',')
# timestamp_data = np.genfromtxt(f'{filepath}timestamp.txt', delimiter=',')
# timestamp_data = timestamp_data[48:-12, :]

filepath = './20200112_notate/'
register_log = np.genfromtxt(f'{filepath}log_all_1.csv',delimiter=',')
register_log_no_correction = np.genfromtxt(f'{filepath}log_no_correction_1.csv',delimiter=',')
register_log_only_loop = np.genfromtxt(f'{filepath}log_only_loop_correction_1.csv',delimiter=',')
timestamp_data = np.genfromtxt(f'{filepath}timestamp.txt', delimiter=',')
timestamp_data = timestamp_data[12:-12, :]

plt.rcParams.update({"font.size" : 40 })
plt.rcParams.update({"font.sans-serif" : "Arial"})
plt.rcParams.update({"font.family" : "sans-serif"})

# fig = plt.figure(1, figsize=(14,10), dpi=100)
# fig.subplots_adjust(bottom=0.2)
# fig.subplots_adjust(top=0.99)
# fig.subplots_adjust(right=0.99)
# singular_value = register_log[1:, 6]
# register_error = register_log[1:, 4]
# plt.scatter(singular_value, register_error, linewidths=5)

# linex = 0.3*np.ones((100,))
# liney = np.linspace(0, 0.2, 100)
# plt.plot(linex, liney, "r", linewidth=5)

# plt.xlabel('$||\mathbf{L\hat{H}}|| $')
# plt.ylabel("Error (rad)")
# plt.yticks(np.arange(0,0.21,step=0.05))
# plt.xticks(np.arange(0,0.8,step=0.1))
# plt.grid()
# # plt.legend()
# plt.savefig("LH_error_precession.pdf")
# plt.show()

# fig = plt.figure(2, figsize=(14,10))
# fig.subplots_adjust(bottom=0.2)
# fig.subplots_adjust(top=0.99)
# fig.subplots_adjust(right=0.99)
# singular_value = register_log[1:, 6]
# time_value = timestamp_data[1:,1]

# plt.plot(time_value, singular_value, linewidth=5)
# liney = 0.3*np.ones((100,))
# linex = np.linspace(1, 120, 100)
# plt.plot(linex, liney, "r", linewidth=5)

# plt.xlabel('Time(s)')
# plt.ylabel('$||\mathbf{L\hat{H}}|| $')
# plt.yticks(np.arange(0,0.75,step=0.1))
# plt.xticks(np.arange(0,121,step=20))
# plt.xlim([0,125])
# plt.grid()
# # plt.legend()
# plt.savefig("LH_precession.pdf")
# plt.show()

fig = plt.figure(3, figsize=(16,10))
fig.subplots_adjust(bottom=0.15)
fig.subplots_adjust(top=0.99)
fig.subplots_adjust(right=0.99)

error1 = register_log[1:, 5]
error2 = register_log_no_correction[1:, 5]
error3 = register_log_only_loop[1:, 5]
time_value = timestamp_data[1:,1]

plt.plot(time_value, error1, "-*", linewidth=4, label="All Corrections")
plt.plot(time_value, error2, "--", linewidth=4, label="No corrections")
plt.plot(time_value, error3, linewidth=4, label="Only Loop Closure Corrections")

plt.xlabel('Time(s)')
plt.ylabel('Attitude Error (rad)')
# plt.yticks(np.arange(0,0.2,step=0.02))
plt.xticks(np.arange(0,121,step=20))
plt.grid()
plt.legend(prop={"size":30})
# plt.savefig("error_3axis.png")
plt.show()
