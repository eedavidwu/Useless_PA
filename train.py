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
#import cv2
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

def train(args,auto_encoder,trainloader,testloader,train_snr):
    #model_name:
    model_name=args.model
    
    # Define an optimizer and criterion
    criterion = nn.MSELoss()
    optimizer = optim.Adam(auto_encoder.parameters(),lr=0.0001)
    
    #Start Train:
    batch_iter=(trainloader.dataset.__len__() // trainloader.batch_size)
    print_iter=int(batch_iter/2)
    best_psnr=0
    epoch_last=0
    best_auto=auto_encoder

    #whether resume:
    if args.resume==True:
        print("Resume")
        #model_path=os.path.join(args.best_ckpt_path,'best_fading_8_transmit_'+str(args.tran_know_flag)+'_equal_1_PA_1_'+model_name+'_symbol_'+str(args.S)+'_SNR_5.pth')
        model_path='./ckpts/best_fading_8_transmit_0_equal_1_PA_0_JSCC_OFDM_SNR_5.pth'
        #model_path='./ckpts/best_fading_8_transmit_1_equal_1_PA_0_JSCC_OFDM_SNR_5.pth'
        #model_path=os.path.join(args.best_ckpt_path,'best_weight_'+model_name+'_SNR_H_'+str(train_snr)+'.pth')
        checkpoint=torch.load(model_path)
        epoch_last=checkpoint["epoch"]
        auto_encoder.load_state_dict(checkpoint["net"])

        optimizer.load_state_dict(checkpoint["op"])
        best_psnr=checkpoint["Ave_PSNR"]
        Trained_SNR=checkpoint['SNR']
        #optimizer=checkpoint['op']

        print("Load model:",model_path)
        print("Model is trained in SNR: ",train_snr," with PSNR:",best_psnr," at epoch ",epoch_last)
        #auto_encoder = auto_encoder.cuda()

    for epoch in range(epoch_last,args.all_epoch):
        auto_encoder.train()
        running_loss = 0.0

        channel_snr=train_snr
        channel_flag=train_snr
        #print('Epoch ',str(epoch),' trained with SNR: ',channel_flag)
        
        for batch_idx, (inputs, _) in enumerate(trainloader, 0):
            inputs = Variable(inputs.cuda())
            # set a random noisy:            
            # ============ Forward ============
            #papr,outputs = auto_encoder(inputs,channel_snr)
            papr=0
            outputs,_ = auto_encoder(inputs,channel_snr)
            loss_mse=criterion(outputs, inputs)
            # ============ Backward ============
            optimizer.zero_grad()
            if args.papr_flag==True:
                #papr_db=10*torch.log10(papr)
                loss_ave_papr=papr.mean()
                #loss_ave_papr=papr_db.mean()
                loss = loss_mse+args.papr_lambda*loss_ave_papr
            else:
                loss = loss_mse

            loss.backward()
            optimizer.step()
            # ============ Ave_loss compute ============
            running_loss += loss.data
            
            if (batch_idx+1) % print_iter == 0:
                if args.papr_flag==True:
                    print("Epoch: [%d] [%4d/%4d] , loss: %.5f, loss_mse: %.5f,  loss_weighted_PAPR: %.5f, loss_PAPR: %.5f" %
                        ((epoch), (batch_idx), (batch_iter),loss,loss_mse ,(args.papr_lambda*loss_ave_papr),loss_ave_papr,))
                else:
                    print("Epoch: [%d] [%4d/%4d] , loss: %.8f" %
                        ((epoch), (batch_idx), (batch_iter), running_loss / print_iter))
                running_loss = 0.0
            
            
        
        if (epoch % 4) ==0:
            ##Validate:
            if args.model=='JSCC_OFDM':
                validate_snr=channel_snr
                ave_psnr=compute_AvePSNR(auto_encoder,testloader,validate_snr)
                print("For Epoch:",epoch,", SNR: ",validate_snr,", get Ave_PSNR:",ave_psnr)

                if ave_psnr > best_psnr:
                    PSNR_list=[]
                    best_psnr=ave_psnr
                    print('############## Find one best model with PSNR:',best_psnr,' under SNR: ',channel_flag)
                    #for i in [1,4,10,16,19]:
                    checkpoint={
                        "model_name":args.model,
                        "net":auto_encoder.state_dict(),
                        "op":optimizer.state_dict(),
                        "epoch":epoch,
                        "SNR":channel_flag,
                        "Ave_PSNR":ave_psnr
                    }
                    save_path=os.path.join(args.best_ckpt_path,'best_rate_8_symbol_'+str(args.S)+'_transmit_'+str(args.tran_know_flag)+'_PA_'+str(args.hard_PA)+'_'+model_name+'_SNR_'+str(channel_snr)+'.pth')
                    torch.save(checkpoint, save_path)
                    print('Saving Model at epoch',epoch,save_path)
          
                 
def compute_AvePSNR(model,dataloader,snr):
    psnr_all_list = []
    model.eval()
    MSE_compute = nn.MSELoss(reduction='none')
    for batch_idx, (inputs, _) in enumerate(dataloader, 0):
        b,c,h,w=inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]
        inputs = Variable(inputs.cuda())
        outputs,_ = model(inputs,snr)
        MSE_each_image = (torch.sum(MSE_compute(outputs, inputs).view(b,-1),dim=1))/(c*h*w)
        PSNR_each_image = 10 * torch.log10(1 / MSE_each_image)
        one_batch_PSNR=PSNR_each_image.data.cpu().numpy()
        psnr_all_list.extend(one_batch_PSNR)
    Ave_PSNR=np.mean(psnr_all_list)
    Ave_PSNR=np.around(Ave_PSNR,5)

    return Ave_PSNR


def main():
    parser = argparse.ArgumentParser()
    #Train:
    parser.add_argument("--best_ckpt_path", default='./ckpts/', type=str,help='best model path')
    parser.add_argument("--all_epoch", default=150, type=int,help='Train_epoch')
    parser.add_argument("--best_choice", default='loss', type=str,help='select epoch [loss/PSNR]')
    parser.add_argument("--flag", default='train', type=str,help='train or eval for JSCC')
    parser.add_argument("--attention_num", default=64, type=int,help='attention_number')

    # Model and Channel:
    parser.add_argument("--model", default='JSCC_OFDM', type=str,help='Model select: DAS_JSCC_OFDM/JSCC_OFDM')
    parser.add_argument("--channel_type", default='awgn', type=str,help='awgn/slow fading/burst')
    parser.add_argument("--h_stddev", default=1, type=float,help='awgn/slow fading/burst')
    parser.add_argument("--equalization",default=1,type=int,help='Equalization_flag 1.eq 2.cat')
    parser.add_argument("--S", default=8, type=int,help='number of symbol')
    parser.add_argument("--M", default=32, type=int,help='number of subcarrier')
    parser.add_argument("--N_pilot", default=1, type=int,help='number of pilot symbol')
    parser.add_argument("--tcn", default=8, type=int,help='tansmit_channel_num for djscc')
    parser.add_argument("--tran_know_flag", default=0, type=int,help='tansmit_know flag')
    parser.add_argument("--hard_PA", default=1, type=int,help='tansmit_PA')
    parser.add_argument("--sorted_flag", default=1, type=int,help='sorted_flag')


    parser.add_argument("--cp_num", default='8', type=int,help='CP num, 0.25*subcariier')
    parser.add_argument("--gama", default='4', type=int,help='time delay constant for multipath fading channel')

    #PAPR loss:
    parser.add_argument("--papr_lambda", default=0.0005,type=float,help='PAPR parameter')
    parser.add_argument("--papr_flag", default=False,type=bool,help='PAPR parameter')
    parser.add_argument("--clip_flag", default=False,type=bool,help='PAPR parameter')


    parser.add_argument("--input_snr_max", default=20, type=float,help='SNR (db)')
    parser.add_argument("--input_snr_min", default=0, type=int,help='SNR (db)')
    parser.add_argument("--train_snr_list",nargs='+', type=int, help='Train SNR (db)')
    #parser.add_argument("--train_snr_list_in",nargs='+', type=list, help='Train SNR (db)')

    parser.add_argument("--train_snr",default=5, type=int, help='Train SNR (db)')

    parser.add_argument("--resume", default=False,type=bool, help='Load past model')
    #parser.add_argument("--snr_num",default=4,type=int,help="num of snr")

    #GPU_ids = [0,1]


    global args
    args=parser.parse_args()

    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                              shuffle=True, num_workers=2,drop_last=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=2,drop_last=True)
    
    # Create model
    # Create model
    print('equalization:',args.equalization)
    print('h_stdev',args.h_stddev)
    print('trainsmitter know:',args.tran_know_flag)
    print('Equal flag:',args.equalization)	
    print('PA_flag:',args.hard_PA)
    print("rate:",args.tcn)
    print("subcarrier:",args.M)
    print("symbols:",args.S)



    if args.model=='JSCC_OFDM':
        auto_encoder=OFDM_models.Classic_JSCC(args)
        #auto_encoder = nn.DataParallel(auto_encoder,device_ids = GPU_ids)
        auto_encoder = auto_encoder.cuda()
        print("Create the model:",args.model)
        #nohup python train.py --train_snr 10 --tran_know_flag 0 --equalization 2 > nohup_unknown_jscc_10.out&  know:26214->unknwo:25833
        train_snr=args.train_snr
        #train_snr=10
        print("############## Train model with SNR: ",train_snr," ##############")
        train(args,auto_encoder,trainloader,testloader,train_snr)
    #nohup python train.py --tran_know_flag 1 --equalization 2 --all_epoch 200 --model JSCC_OFDM --resume True --train_snr 10 > nohup_8_known_jscc_10.out&
    
if __name__ == '__main__':
    main()
