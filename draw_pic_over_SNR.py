import matplotlib.pyplot as plt
import numpy as np

x=range(20)

SNR_5=[23.06252, 24.2463, 25.33498, 26.26576, 27.09694, 27.77159, 28.30266, 28.70132, 28.986, 29.1673, 29.25914, 29.29236, 29.28406, 29.23716, 29.17372, 29.10388, 29.01975, 28.94755, 28.86807, 28.79723]
SNR_10=[21.30101, 22.47974, 23.63021, 24.75836, 25.79884, 26.80265, 27.69012, 28.501, 29.23182, 29.86758, 30.40595, 30.8564, 31.23364, 31.52115, 31.7421, 31.91446, 32.03954, 32.12212, 32.17788, 32.21594]

SNR_15=[19.99406, 21.02712, 22.07203, 23.13978, 24.1623, 25.17192, 26.13398, 27.04476, 27.90211, 28.69346, 29.43338, 30.11939, 30.7267, 31.27762, 31.76604, 32.18795, 32.56032, 32.87568, 33.14435, 33.35913]

SNR_19=[19.32804, 20.34911, 21.34226, 22.35544, 23.35908, 24.33581, 25.3032, 26.23549, 27.1132, 27.97523, 28.76901, 29.52731, 30.21174, 30.83554, 31.42259, 31.95041, 32.41034, 32.81802, 33.16658, 33.48492]
SNR_attention=[24.21228, 25.03544, 25.82678, 26.56092, 27.28298, 27.94467, 28.57928, 29.16796, 29.71776, 30.2278, 30.69616, 31.12796, 31.5279, 31.89536, 32.21882, 32.51213, 32.77456, 32.99307, 33.18502, 33.33776]
SNR_attention_trans_unknown=[24.06394, 24.97695, 25.80042, 26.58552, 27.31134, 27.99362, 28.61556, 29.20681, 29.75499, 30.26839, 30.73644, 31.18264, 31.58056, 31.94388, 32.27398, 32.57256, 32.82736, 33.0532, 33.25692, 33.43008]
plt.title('Performance of different models')
plt.xlabel('SNR (dB)', size=10)
plt.ylabel('Ave_PSNR (dB)', size=10)
plt.plot(x, SNR_5, color='black', linestyle='-', marker='o', label='Traditional JSCC tranined in SNR 5')
plt.plot(x, SNR_10, color='g', linestyle='-', marker='*',label='Traditional JSCC tranined in SNR 10')
plt.plot(x, SNR_15, color='b', linestyle='-', marker='*',label='Traditional JSCC tranined in SNR 15')
plt.plot(x, SNR_19, color='grey', linestyle='-', marker='*',label='Traditional JSCC tranined in SNR 19')
plt.plot(x, SNR_attention, color='r', linestyle='-', marker='*',label='Attention JSCC')
plt.plot(x, SNR_attention_trans_unknown, color='pink', linestyle='-', marker='*',label='Attention JSCC without transmitter')


plt.legend()
plt.show()
plt.savefig('./PSNR_OFDM_attention.jpg')

