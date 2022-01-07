import matplotlib.pyplot as plt
import numpy as np

x=range(20)
###8:

trans_ours=[24.7323, 25.4537, 26.0907, 26.6417, 27.0871, 27.4651, 27.7428, 27.972, 28.1333, 28.2542, 28.3412, 28.3936, 28.4319, 28.4556, 28.4716, 28.474, 28.4796, 28.474, 28.4727, 28.4662]
trans_know=[21.7259, 22.581, 23.3522, 24.0469, 24.6095, 25.0724, 25.44, 25.7274, 25.9546, 26.1107, 26.2474, 26.3345, 26.4116, 26.4662, 26.5073, 26.5337, 26.5651, 26.5905, 26.6016, 26.6091]
trans_unknow=[22.1978, 23.0771, 23.8516, 24.5302, 25.0828, 25.5235, 25.8552, 26.1151, 26.2969, 26.4352, 26.5369, 26.6079, 26.6674, 26.7107, 26.7332, 26.7558, 26.7804, 26.7959, 26.8, 26.8066]


###16:
#trans_ours=[21.30101, 22.47974, 23.63021, 24.75836, 25.79884, 26.80265, 27.69012, 28.501, 29.23182, 29.86758, 30.40595, 30.8564, 31.23364, 31.52115, 31.7421, 31.91446, 32.03954, 32.12212, 32.17788, 32.21594]
#trans_know=[25.5355, 26.3616, 27.0691, 27.669, 28.1494, 28.5376, 28.8357, 29.0599, 29.2414, 29.3742, 29.4879, 29.5591, 29.6292, 29.676, 29.7056, 29.7389, 29.7635, 29.7908, 29.7946, 29.8082]
#trans_unknow=[25.9615, 26.8526, 27.6028, 28.2137, 28.6975, 29.0648, 29.3367, 29.5352, 29.6924, 29.7992, 29.8864, 29.938, 29.9933, 30.0266, 30.0492, 30.069, 30.0811, 30.1033, 30.1082, 30.1182]
#(8)
plt.title('Performance of different models')
plt.xlabel('SNR (dB)', size=10)
plt.ylabel('Ave_PSNR (dB)', size=10)
plt.plot(x, trans_know, color='black', linestyle='-', marker='o', label='trans_unknow')
plt.plot(x, trans_unknow, color='g', linestyle='-', marker='*',label='trans_know')
plt.plot(x, trans_ours, color='r', linestyle='-', marker='*',label='trans_ours')

plt.legend()
plt.show()
plt.savefig('./PSNR_Origiinal_OFDM_trans_8_10.jpg')

