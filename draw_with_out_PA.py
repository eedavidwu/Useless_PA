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

inner_mse_0=[22.73057, 16.75769, 13.77103, 17.90975, 10.50352, 6.65097, 6.06636, 5.85188, 3.6118, 3.69551, 2.96854]
inner_mse_1=[1.79936, 1.56231, 1.17386, 0.85006, 0.6521, 0.5576, 0.48441, 0.39221, 0.34149, 0.26037, 0.19446, 0.14459, 0.11991, 0.08647, 0.08235, 0.05611, 0.05247, 0.03529, 0.02922]

out_mse_0=[0.01119, 0.00923, 0.0076, 0.00629, 0.00524, 0.0044, 0.00376, 0.00325, 0.00286, 0.00256, 0.00234,]
out_mse_1=[ 0.00217, 0.00204, 0.00195, 0.00189, 0.00184, 0.00181, 0.00179, 0.00178, 0.00177, 0.00177, 0.00177, 0.00177, 0.00177, 0.00178, 0.00178, 0.00178, 0.00178, 0.00179, 0.00179]

##without PA sorted:
#in mse list: [22.50932, 15.37422, 16.00354, 9.14494, 8.69816, 6.83961, 5.29952, 4.3792, 2.91827, 2.64917, 2.24627, 1.64026, 1.53754, 0.97285, 0.94082, 0.68522, 0.70818, 1.38874, 0.54184, 0.33132, 0.29578, 0.44755, 0.14216, 0.10665, 0.0815, 0.06133, 0.05101, 0.03884, 0.03078, 0.03327]
#out mse list: [0.00659, 0.00555, 0.0047, 0.00398, 0.00341, 0.00295, 0.00258, 0.00229, 0.00207, 0.0019, 0.00178, 0.00169, 0.00162, 0.00158, 0.00155, 0.00154, 0.00153, 0.00153, 0.00153, 0.00154, 0.00155, 0.00156, 0.00156, 0.00157, 0.00158, 0.00158, 0.00159, 0.0016, 0.0016, 0.0016]
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
ax3.axis([0,21,0,1.85])
ax3.invert_xaxis()
ax3.set(xlabel='SNR (dB)',ylabel='PA_mse')
ax3.set_title("c). Inner mse [0 to 20 dB]")



ax4.plot(x_0, inner_mse_0, color='b', linestyle='-', marker='*')
ax4.axis([-11,1,0,20])
ax4.invert_xaxis()
ax4.set(xlabel='SNR (dB)',ylabel='PA_mse')
ax4.set_title("d). Inner mse [-10 to 0 dB]")


plt.savefig('./perfect_h_no_PA_in_four_fig.jpg')
'''
fig = plt.figure()

inner_all=[22.73057, 16.75769, 13.77103, 17.90975, 10.50352, 6.65097, 6.06636, 5.85188, 3.6118, 3.69551, 2.96854,1.79936, 1.56231, 1.17386, 0.85006, 0.6521, 0.5576, 0.48441, 0.39221, 0.34149, 0.26037, 0.19446, 0.14459, 0.11991, 0.08647, 0.08235, 0.05611, 0.05247, 0.03529, 0.02922]
out_all=[0.01119, 0.00923, 0.0076, 0.00629, 0.00524, 0.0044, 0.00376, 0.00325, 0.00286, 0.00256, 0.00234,0.00217, 0.00204, 0.00195, 0.00189, 0.00184, 0.00181, 0.00179, 0.00178, 0.00177, 0.00177, 0.00177, 0.00177, 0.00177, 0.00178, 0.00178, 0.00178, 0.00178, 0.00179, 0.00179]

x_2=np.arange(-10,20,1)

plt.plot(x_2, out_all, color='r', linestyle='-', marker='o', label='out_mse')
plt.plot(x_2, inner_all, color='b', linestyle='-', marker='*',label='PA_mse')
#ax = plt.axes()
#ax.invert_xaxis()
plt.legend()
#plt.xlabel('SNR (dB)', size=20)
#plt.ylabel('mse', size=20)
plt.savefig('./perfect_h_no_PA_in_one_fig.jpg')

'''
