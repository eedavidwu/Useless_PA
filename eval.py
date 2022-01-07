import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import model.OFDM as OFDM_models
import os
import argparse
from random import randint
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

          
                 
def compute_AvePSNR(model,dataloader,snr):
    psnr_all_list = []
    model.eval()
    MSE_compute = nn.MSELoss(reduction='none')
    for batch_idx, (inputs, _) in enumerate(dataloader, 0):
        b,c,h,w=inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]
        inputs = Variable(inputs.cuda())
        outputs = model(inputs,snr)
        MSE_each_image = (torch.sum(MSE_compute(outputs, inputs).view(b,-1),dim=1))/(c*h*w)
        PSNR_each_image = 10 * torch.log10(1 / MSE_each_image)
        one_batch_PSNR=PSNR_each_image.data.cpu().numpy()
        psnr_all_list.extend(one_batch_PSNR)
    Ave_PSNR=np.mean(psnr_all_list)
    Ave_PSNR=np.around(Ave_PSNR,4)

    return Ave_PSNR

def compute_Mse(model,dataloader,snr):
    out_mse_list = []
    inner_mse_list=[]
    model.eval()
    for batch_idx, (inputs, _) in enumerate(dataloader, 0):
        b,c,h,w=inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]
        inputs = Variable(inputs.cuda())
        outputs,inner_mse = model(inputs,snr)
        out_mse=torch.nn.functional.mse_loss(outputs, inputs)
        one_batch_inner_mse=inner_mse.view(1).data.cpu().numpy()
        one_batch_out_mse=out_mse.view(1).data.cpu().numpy()
        out_mse_list.extend(one_batch_out_mse)
        inner_mse_list.extend(one_batch_inner_mse)

    out_mse_mean=np.mean(out_mse_list)
    in_mse_mean=np.mean(inner_mse_list)


    return in_mse_mean,out_mse_mean,

def main():
    parser = argparse.ArgumentParser()
    #Train:
    parser.add_argument("--best_ckpt_path", default='./ckpts/', type=str,help='best model path')
    parser.add_argument("--all_epoch", default='200', type=int,help='Train_epoch')
    parser.add_argument("--best_choice", default='loss', type=str,help='select epoch [loss/PSNR]')
    parser.add_argument("--flag", default='train', type=str,help='train or eval for JSCC')
    parser.add_argument("--attention_num", default=0, type=int,help='attention_number')

    # Model and Channel:
    parser.add_argument("--model", default='JSCC_OFDM', type=str,help='Model select: DAS_JSCC_OFDM/JSCC_OFDM/DAS_ALL_JSCC_OFDM')
    parser.add_argument("--channel_type", default='awgn', type=str,help='awgn/slow fading/burst')
    #parser.add_argument("--const_snr", default=True,help='SNR (db)')
    #parser.add_argument("--input_const_snr", default=1, type=float,help='SNR (db)')
    parser.add_argument("--cp_num", default='16', type=int,help='CP num, 0.25*subcariier')
    parser.add_argument("--gama", default='4', type=int,help='time delay constant for multipath fading channel')
    #PAPR loss:
    parser.add_argument("--h_stddev", default=2, type=float,help='awgn/slow fading/burst')
    parser.add_argument("--equalization", default=1, type=float,help='Equalization_flag')
    parser.add_argument("--S", default=8, type=int,help='number of symbol')
    parser.add_argument("--M", default=32, type=int,help='number of subcarrier')
    parser.add_argument("--tcn", default=8, type=int,help='tansmit_channel_num for djscc')
    parser.add_argument("--N_pilot", default=1, type=int,help='number of pilot symbol')

    parser.add_argument("--input_snr_max", default=20, type=float,help='SNR (db)')
    parser.add_argument("--input_snr_min", default=0, type=int,help='SNR (db)')
    parser.add_argument("--train_snr_list",nargs='+', type=int, help='Train SNR (db)')
    #parser.add_argument("--train_snr_list_in",nargs='+', type=list, help='Train SNR (db)')
    parser.add_argument("--tran_know_flag", default=0,type=int,help='tansmit_know flag')
    parser.add_argument("--hard_PA", default=0, type=int,help='tansmit_PA')
    parser.add_argument("--sorted_flag", default=1, type=int,help='sorted_flag')

    parser.add_argument("--train_snr",default=5, type=int, help='Train SNR (db)')

    parser.add_argument("--resume", default=False,type=bool, help='Load past model')
    #parser.add_argument("--snr_num",default=4,type=int,help="num of snr")

    #GPU_ids = [0,1,2,3]

    global args
    args=parser.parse_args()

    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=2)
    #model_name='JSCC_OFDM'
    #train_snr=5
    ckpt_folder='./ckpts'

    if args.model=='DAS_JSCC_OFDM':
        train_snr='random'
        auto_encoder=OFDM_models.Attention_all_JSCC(args)
        model_path=os.path.join(ckpt_folder,'best_fading_transmit_'+str(args.tran_know_flag)+'_equal_'+str(args.equalization)+args.model+'_SNR_'+str(train_snr)+'.pth')

        #python eval.py --tran_know_flag 1 --model DAS_JSCC_OFDM

    elif args.model=='JSCC_OFDM':
        train_snr=args.train_snr
        auto_encoder=OFDM_models.Classic_JSCC(args)
        model_path='./ckpts/best_rate_8_symbol_8_transmit_0_PA_0_sorted_JSCC_OFDM_SNR_5.pth'
        #model_path='./ckpts/best_rate_8_symbol_8_transmit_0_PA_1_JSCC_OFDM_SNR_5.pth'

    #for train_snr in ['5','10','15','19']:
        #train_snr='5'
        #train_snr='random_in_list'
        #Create model
    #auto_encoder = nn.DataParallel(auto_encoder,device_ids = GPU_ids)
    auto_encoder = auto_encoder.cuda()
    print("Create the model:",args.model)

    checkpoint=torch.load(model_path)
    epoch_last=checkpoint["epoch"]
    auto_encoder.load_state_dict(checkpoint["net"])

    best_psnr=checkpoint["Ave_PSNR"]
    Trained_SNR=checkpoint['SNR']

    print("Load model:",model_path)
    print("Model is trained in SNR: ",Trained_SNR," with PSNR:",best_psnr," at epoch ",epoch_last)


    #./ckpts/best_fading_8_transmit_0_equal_2_JSCC_OFDM_SNR_10.pth


    #PSNR_list=[[0 for a in range(6)] for b in range(6)]
    #SNR_1_list=[1,4,9,12,16,19]
    #SNR_2_list=[1,4,9,12,16,19]
    PSNR_list=[]
    in_mse_list=[]
    out_mse_list=[]


    for i in np.arange(-10,20,1):
        #i=4
        #j=2
        validate_snr=i
        #one_ave_psnr=compute_AvePSNR(auto_encoder,testloader,validate_snr)
        #print("Compute Ave_PSNR in SNR:",validate_snr,"with PSNR:",one_ave_psnr)
        #PSNR_list.append(one_ave_psnr)
        in_mse_ave_each_snr=0
        out_mse_ave_each_snr=0
        test_time=10
        for j in range(test_time):
            in_mse_test,out_mse_test=compute_Mse(auto_encoder,testloader,validate_snr)
            in_mse_ave_each_snr=in_mse_ave_each_snr+in_mse_test
            out_mse_ave_each_snr=out_mse_ave_each_snr+out_mse_test
        out_mse_ave_each_snr=out_mse_ave_each_snr/test_time
        in_mse_ave_each_snr=in_mse_ave_each_snr/test_time
        out_mse_ave_each_snr=np.around(out_mse_ave_each_snr,5)
        in_mse_ave_each_snr=np.around(in_mse_ave_each_snr,5)

        print("Compute Mse in SNR:",validate_snr,"with in-mse:",in_mse_ave_each_snr,' with out-mse:',out_mse_ave_each_snr)
        in_mse_list.append(in_mse_ave_each_snr)
        out_mse_list.append(out_mse_ave_each_snr)

    print("Finish validate",model_path)  
    print('in mse list:',in_mse_list)
    print('out mse list:',out_mse_list)
    #print(PSNR_list)  
if __name__ == '__main__':
    main()
