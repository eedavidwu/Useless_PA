import matplotlib.pyplot as plt
import numpy as np

x=np.arange(-10,1,1)

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax3 = fig.add_subplot(2,2,3)

ax2 = fig.add_subplot(2,2,2)
ax4 = fig.add_subplot(2,2,4)
plt.subplots_adjust(wspace=0.4,hspace=0.45)

x_0=np.arange(-10,1,1)
x_1=np.arange(1,20,1)
inner_mse_0=[1.98471, 1.73466, 1.53569, 1.3769, 1.25141, 1.15155, 1.07192, 1.00874, 0.95869, 0.91893, 0.88739]
inner_mse_1=[0.86234, 0.84244, 0.827, 0.81433, 0.80435, 0.7959, 0.78991, 0.78441, 0.78098, 0.77775, 0.77545, 0.77358, 0.77153, 0.77038, 0.76975, 0.7686, 0.76812, 0.76744, 0.76727]

out_mse_0=[0.00988, 0.00879, 0.0079, 0.00718, 0.00662, 0.00619, 0.00587, 0.00562, 0.00545, 0.00532, 0.00524]
out_mse_1=[0.00518, 0.00514, 0.00512, 0.00511, 0.0051, 0.00509, 0.00509, 0.00509, 0.0051, 0.0051, 0.0051, 0.0051, 0.0051, 0.0051, 0.00511, 0.00511, 0.00511, 0.00511, 0.00511]

ax1.plot(x_1, out_mse_1, color='r', linestyle='-', marker='o')
ax1.axis([0,21,0,0.02])
ax1.invert_xaxis()
ax1.set(xlabel='SNR (dB)',ylabel='Out_mse')
ax1.set_title("a). Out mse [0 to 20 dB]")


ax2.plot(x_0, out_mse_0, color='r', linestyle='-', marker='o')
ax2.axis([-11,1,0,0.02])
ax2.invert_xaxis()
ax2.set(xlabel='SNR (dB)',ylabel='Out_mse')
ax2.set_title("b). Out mse [-10 to 0 dB]")


ax3.plot(x_1, inner_mse_1, color='b', linestyle='-', marker='*')
ax3.axis([0,21,0,5])
ax3.invert_xaxis()
ax3.set(xlabel='SNR (dB)',ylabel='PA_mse')
ax3.set_title("c). Inner mse [0 to 20 dB]")


ax4.plot(x_0, inner_mse_0, color='b', linestyle='-', marker='*')
ax4.axis([-11,1,0,5])
ax4.invert_xaxis()
ax4.set(xlabel='SNR (dB)',ylabel='PA_mse')
ax4.set_title("d). Inner mse [-10 to 0 dB]")

plt.savefig('./perfect_h_PA_in_four_fig.jpg')
