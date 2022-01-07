import os
from re import X
from torch.functional import split
import torch.nn.functional as F
import  torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import random


class Fading_Channel(nn.Module):
    def __init__(self, args):
        super(Fading_Channel, self).__init__()
        self.gama=args.gama

    def complex_convolution(self,inputs,h):
        in_shape=inputs.shape
        sub_channel_num=in_shape[2]
        kernel_real=torch.real(h)
        kernel_img=torch.imag(h)


        weight_real=nn.Parameter(data=kernel_real,requires_grad=False)
        weight_img=nn.Parameter(data=kernel_img,requires_grad=False)

        inputs_real=torch.real(inputs).float()
        inputs_img=torch.imag(inputs).float()

        out_r_r=F.conv1d(inputs_real,weight_real,groups=12,padding=sub_channel_num)[:,:,sub_channel_num-sub_channel_num//2:sub_channel_num+sub_channel_num//2,]
        out_r_im=F.conv1d(inputs_real,weight_img,groups=12,padding=sub_channel_num)[:,:,sub_channel_num-sub_channel_num//2:sub_channel_num+sub_channel_num//2,]
        out_im_r=F.conv1d(inputs_img,weight_real,groups=12,padding=sub_channel_num)[:,:,sub_channel_num-sub_channel_num//2:sub_channel_num+sub_channel_num//2,]
        out_im_im=F.conv1d(inputs_img,weight_img,groups=12,padding=sub_channel_num)[:,:,sub_channel_num-sub_channel_num//2:sub_channel_num+sub_channel_num//2,]
        out_real=out_r_r-out_im_im
        out_im=out_r_im+out_im_r
        out=torch.complex(out_real,out_im)
        return out,h
    def compensation(self,inputs,h_broadcast):
        h_norm_square_broadcast=torch.square(torch.abs(h_broadcast))
        h_conj_broadcast=torch.conj(h_broadcast)
        h_norm_square_inverse=torch.where(h_norm_square_broadcast>0,1/h_norm_square_broadcast,h_norm_square_broadcast)

        compensate_matrix=torch.transpose(h_norm_square_inverse*h_conj_broadcast,1,2)
        out=torch.bmm(compensate_matrix,inputs)
        #out=(h_conj_broadcast*inputs)/h_norm_broadcast
        return out
        
    def forward(self, inputs,h,input_snr):
        in_shape=inputs.shape
        batch_size=in_shape[0]

        ##Through fading channel
        complex_para_out=torch.bmm(h,inputs)

        ##awgn:
        noise_stddev=(np.sqrt(10**(-input_snr/10))/np.sqrt(2)).reshape(-1,1,1)
        noise_stddev_board=torch.from_numpy(noise_stddev).repeat(1,in_shape[1],in_shape[2]).cuda()
        mean=torch.zeros_like(noise_stddev_board).cuda()
        noise_real=Variable(torch.normal(mean=mean,std=noise_stddev_board).cuda()).float()
        noise_img=Variable(torch.normal(mean=mean,std=noise_stddev_board).cuda()).float()
        noise_complex=torch.complex(noise_real,noise_img)

        #idea H:
        H_idea=torch.ones_like(h)
        
        ##y=hx+w
        channel_out=complex_para_out+noise_complex

        ##compensation:
        channel_com_out=self.compensation(channel_out,h)

        new_noise=self.compensation(noise_complex,h)
        new_out=torch.bmm(H_idea,inputs)+new_noise

        return channel_com_out#,inputs_in_norm

class FL_En_Module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride,padding,activation=None):
        super(FL_En_Module, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding)
        self.GDN = nn.BatchNorm2d(out_channels)
        if activation=='sigmoid':
            self.activate_func=nn.Sigmoid()
        elif activation=='prelu':
            self.activate_func=nn.PReLU()
        elif activation==None:
            self.activate_func=None            

    def forward(self, inputs):
        out_conv1=self.conv1(inputs)
        out_bn=self.GDN(out_conv1)
        if self.activate_func != None:
            out=self.activate_func(out_bn)
        else:
            out=out_bn
        return out
'''
class AL_Module(nn.Module):
    def __init__(self,fc_in):
        super(AL_Module, self).__init__()
        self.Ave_Pooling = nn.AdaptiveAvgPool2d(1)
        self.FC_1 = nn.Linear(fc_in+65,fc_in//4)
        self.FC_2 = nn.Linear(fc_in//4,fc_in)

    def forward(self, inputs,h,snr):
        out_pooling=self.Ave_Pooling(inputs).squeeze()
        b=inputs.shape[0]
        c=inputs.shape[1]
        in_fc=torch.cat((snr,h,out_pooling),dim=1).float()
        out_fc_1=self.FC_1(in_fc)
        out_fc_1_relu=torch.nn.functional.relu(out_fc_1)
        out_fc_2=self.FC_2(out_fc_1_relu)
        out_fc_2_sig=torch.sigmoid(out_fc_2).view(b,c,1,1)
        out=out_fc_2_sig*inputs
        return out
'''
class AL_En_Module(nn.Module):
    def __init__(self,attention_size,channel_in_size,feature_in_size):
        super(AL_En_Module, self).__init__()
        self.Channel_attention = AL_CH_Module(attention_size,channel_in_size)
        self.Spatial_attention = AL_SP_Module(attention_size,channel_in_size,feature_in_size)

    def forward(self, inputs,attention):
        #attention
        c_out_=self.Channel_attention(inputs,attention)
        s_out=self.Spatial_attention(c_out_,attention)
        return s_out+inputs
class AL_De_Module(nn.Module):
    def __init__(self,attention_size,channel_in_size,feature_in_size):
        super(AL_De_Module, self).__init__()
        self.Channel_attention = AL_CH_Module(attention_size,channel_in_size)
        self.Spatial_attention = AL_SP_Module(attention_size,channel_in_size,feature_in_size)

    def forward(self, inputs,attention):
        #attention=
        #b=h.shape[0]
        #sub=h.shape[1]
        #h=h.unsqueeze(dim=2)
        #snr=snr.repeat(1,sub).unsqueeze(dim=2)
        b=inputs.shape[0]
        attention=attention.view(b,-1)
        c_out=self.Channel_attention(inputs,attention)
        s_out=self.Spatial_attention(c_out,attention)
        return s_out

class AL_CH_Module(nn.Module):
    def __init__(self,attention_size,channel_size):
        super(AL_CH_Module, self).__init__()
        self.Ave_Pooling = nn.AdaptiveAvgPool2d(1)
        self.FC_1 = nn.Linear(channel_size+attention_size,channel_size//16)
        self.FC_2 = nn.Linear(channel_size//16,channel_size)

    def forward(self, inputs,attention):
        out_pooling=self.Ave_Pooling(inputs).squeeze()
        b=inputs.shape[0]
        c=inputs.shape[1]
        in_fc=torch.cat((attention,out_pooling),dim=1).float()
        out_fc_1=self.FC_1(in_fc)
        out_fc_1_relu=torch.nn.functional.relu(out_fc_1)
        out_fc_2=self.FC_2(out_fc_1_relu)
        out_fc_2_sig=torch.sigmoid(out_fc_2).view(b,c,1,1)
        out=out_fc_2_sig*inputs+inputs
        return out

class AL_SP_Module(nn.Module):
    def __init__(self,attention_size,channel_in_size,feature_in_size):
        super(AL_SP_Module, self).__init__()
        self.channel_pooling= nn.Conv2d(channel_in_size, 1, kernel_size=5, stride=1,padding=2)

        #self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding)
        #self.Ave_Pooling = nn.AdaptiveAvgPool2d(1)
        #self.FC_1 = nn.Linear(65,fc_in)
        self.FC_1 = nn.Linear(feature_in_size+attention_size,feature_in_size//16)
        self.FC_2 = nn.Linear(feature_in_size//16,feature_in_size)

    def forward(self, inputs,attention):
        out_channel_pooling=self.channel_pooling(inputs).squeeze()
        b=inputs.shape[0]
        feature_size=inputs.shape[2]
        out_pooling=out_channel_pooling.view(b,-1)

        in_fc=torch.cat((attention,out_pooling),dim=1).float()
        out_fc_1=self.FC_1(in_fc)
        out_fc_1_relu=torch.nn.functional.relu(out_fc_1)
        out_fc_2=self.FC_2(out_fc_1_relu)
        out_fc_2_sig=torch.sigmoid(out_fc_2).view(b,1,feature_size,feature_size)
        out=out_fc_2_sig*inputs+inputs
        return out

class PA_En_Module(nn.Module):
    def __init__(self,attention_size,nun_sub):
        super(PA_En_Module, self).__init__()
        self.FC_1 = nn.Linear(attention_size,attention_size//16)
        self.FC_2 = nn.Linear(attention_size//16,nun_sub)
    def forward(self, h):
        out_fc_1=self.FC_1(h)
        out_fc_1_relu=torch.nn.functional.relu(out_fc_1)
        out_fc_2=self.FC_2(out_fc_1_relu)
        out_fc_2_re=torch.nn.functional.relu(out_fc_2)
        return out_fc_2_re

class PA_De_Module(nn.Module):
    def __init__(self,attention_size,nun_sub):
        super(PA_De_Module, self).__init__()
        self.FC_1 = nn.Linear(attention_size,attention_size//16)
        self.FC_2 = nn.Linear(attention_size//16,nun_sub)
    def forward(self, h):
        out_fc_1=self.FC_1(h)
        out_fc_1_relu=torch.nn.functional.relu(out_fc_1)
        out_fc_2=self.FC_2(out_fc_1_relu)
        out_fc_2_re=torch.nn.functional.relu(out_fc_2)
        return out_fc_2_re

class FL_De_Module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride,padding,out_padding,activation=None):
        super(FL_De_Module, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding,output_padding=out_padding)
        self.GDN = nn.BatchNorm2d(out_channels)
        if activation=='sigmoid':
            self.activate_func=nn.Sigmoid()
        elif activation=='prelu':
            self.activate_func=nn.PReLU()
        elif activation==None:
            self.activate_func=None            

    def forward(self, inputs):
        out_deconv1=self.deconv1(inputs)
        out_bn=self.GDN(out_deconv1)
        if self.activate_func != None:
            out=self.activate_func(out_bn)
        else:
            out=out_bn
        return out

class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.FL_Module_1 = FL_En_Module(3, 256, 9, 2, 4,'prelu')
        self.FL_Module_2 = FL_En_Module(256, 256, 5, 2,2, 'prelu')
        self.FL_Module_3 = FL_En_Module(256, 256, 5, 1,2, 'prelu')
        self.FL_Module_4 = FL_En_Module(256, 256, 5, 1,2, 'prelu')
        self.FL_Module_5 = FL_En_Module(256, args.tcn, 5, stride=1,padding=2)

    def forward(self, x):
        batch_size=x.shape[0]
        encoded_1_out = self.FL_Module_1(x)
        encoded_2_out = self.FL_Module_2(encoded_1_out)
        encoded_3_out = self.FL_Module_3(encoded_2_out)
        encoded_4_out = self.FL_Module_4(encoded_3_out)
        encoded_5_out = self.FL_Module_5(encoded_4_out)
        return encoded_5_out

class Decoder(nn.Module):
    def __init__(self,args):
        super(Decoder, self).__init__()
        self.FL_De_Module_1 = FL_De_Module(args.tcn, 256, 5, stride=1,padding=2,out_padding=0,activation='prelu')
        self.FL_De_Module_2 = FL_De_Module(256, 256, 5, stride=1,padding=2,out_padding=0,activation='prelu')
        self.FL_De_Module_3 = FL_De_Module(256, 256, 5, stride=1,padding=2,out_padding=0,activation='prelu')
        self.FL_De_Module_4 = FL_De_Module(256, 256, 5, stride=2,padding=2,out_padding=1,activation='prelu')
        self.FL_De_Module_5 = FL_De_Module(256,3, 9, stride=2,padding=4,out_padding=1,activation='sigmoid')


    def forward(self, x):
        #make the input for decoder:
        #x=torch.cat((Y_p,Y_p_head,Y_head),dim=1).view(encoded_shape[0],-1,encoded_shape[2],encoded_shape[3])

        decoded_1_out = self.FL_De_Module_1(x)
        decoded_2_out = self.FL_De_Module_2(decoded_1_out)
        decoded_3_out = self.FL_De_Module_3(decoded_2_out)
        decoded_4_out = self.FL_De_Module_4(decoded_3_out)
        decoded_5_out = self.FL_De_Module_5(decoded_4_out)
        return decoded_5_out

class Attention_Encoder(nn.Module):
    def __init__(self,args,attention_size):
        super(Attention_Encoder, self).__init__()
        num_subcarrier=args.M
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.FL_Module_1 = FL_En_Module(3, 256, 9, 2, 4,'prelu')
        #[b,256,16,16]
        #self.AL_Module_1=AL_En_Module(attention_size,256,256)
        
        self.FL_Module_2 = FL_En_Module(256, 256, 5, 2,2, 'prelu')
        #[b,256,8,8]
        self.AL_Module_2=AL_En_Module(attention_size,256,64)
        
        self.FL_Module_3 = FL_En_Module(256, 256, 5, 1,2, 'prelu')
        #self.AL_Module_3=AL_En_Module(attention_size,256,64)
        #self.AL_CH_Module_3=AL_CH_Module(256)
        #self.AL_SP_Module_3=AL_SP_Module(256,64)

        self.FL_Module_4 = FL_En_Module(256, 256, 5, 1,2, 'prelu')
        self.AL_Module_4=AL_En_Module(attention_size,256,64)
        #self.AL_CH_Module_4=AL_CH_Module(256)
        #self.AL_SP_Module_4=AL_SP_Module(256,64)

        self.FL_Module_5 = FL_En_Module(256, args.tcn, 5, stride=1,padding=2)
        #self.SP_Mask_Module_5=Channel_SP_Mask_Module(attention_size,args.tcn,64)


    def forward(self, x,attention):
        encoded_1_out = self.FL_Module_1(x)
        #attention_encoder_1_out=self.AL_Module_1(encoded_1_out,attention)
        attention_encoder_1_out=encoded_1_out
        encoded_2_out = self.FL_Module_2(attention_encoder_1_out)
        attention_encoder_2_out=self.AL_Module_2(encoded_2_out,attention)
        
        encoded_3_out = self.FL_Module_3(attention_encoder_2_out)
        #attention_encoder_3_out=self.AL_Module_3(encoded_3_out,attention)
        attention_encoder_3_out=encoded_3_out
        
        encoded_4_out = self.FL_Module_4(attention_encoder_3_out)
        attention_encoder_4_out=self.AL_Module_4(encoded_4_out,attention)

        encoded_5_out = self.FL_Module_5(attention_encoder_4_out)

        return encoded_5_out

class Attention_Decoder(nn.Module):
    def __init__(self,args,attention_size):
        super(Attention_Decoder, self).__init__()
        self.FL_De_Module_1 = FL_De_Module(args.tcn, 256, 5, stride=1,padding=2,out_padding=0,activation='prelu')
        #self.AL_De_module_1=AL_De_Module(attention_size,256,64)

        self.FL_De_Module_2 = FL_De_Module(256, 256, 5, stride=1,padding=2,out_padding=0,activation='prelu')
        self.AL_De_module_2=AL_De_Module(attention_size,256,64)

        self.FL_De_Module_3 = FL_De_Module(256, 256, 5, stride=1,padding=2,out_padding=0,activation='prelu')
        #self.AL_De_module_3=AL_De_Module(attention_size,256,64)

        self.FL_De_Module_4 = FL_De_Module(256, 256, 5, stride=2,padding=2,out_padding=1,activation='prelu')
        self.AL_De_module_4=AL_De_Module(attention_size,256,256)

        self.FL_De_Module_5 = FL_De_Module(256,3, 9, stride=2,padding=4,out_padding=1,activation='sigmoid')


    def forward(self, x,attention):
        decoded_1_out = self.FL_De_Module_1(x)
        #attention_decoder_1_out=self.AL_De_module_1(decoded_1_out,attention)
        attention_decoder_1_out=decoded_1_out

        decoded_2_out = self.FL_De_Module_2(attention_decoder_1_out)
        attention_decoder_2_out=self.AL_De_module_2(decoded_2_out,attention)

        decoded_3_out = self.FL_De_Module_3(attention_decoder_2_out)
        #attention_decoder_3_out=self.AL_De_module_3(decoded_3_out,attention)
        attention_decoder_3_out=decoded_3_out

        decoded_4_out = self.FL_De_Module_4(attention_decoder_3_out)
        attention_decoder_4_out=self.AL_De_module_4(decoded_4_out,attention)

        decoded_5_out = self.FL_De_Module_5(attention_decoder_4_out)

        return decoded_5_out

class Classic_JSCC(nn.Module):
    def __init__(self,args):
        super(Classic_JSCC, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        attention_size=args.M
        self.tran_know_flag=args.tran_know_flag
        self.PA_flag=args.hard_PA
        if self.PA_flag==1:
            self.En_PA=PA_En_Module(attention_size*args.S,args.M)
            self.De_PA=PA_De_Module(attention_size*args.S,args.M)
        if self.tran_know_flag==1:
            self.attention_encoder = Attention_Encoder(args,attention_size)
            self.attention_decoder = Attention_Decoder(args,attention_size)
            #self.decoder=Decoder(args)
        else:
            self.encoder = Encoder(args)
            self.decoder=Decoder(args)
        self.num_sub=args.M
        self.num_sym=args.S
        self.sorted=args.sorted_flag
        #pilot_path=args.pilot_path
        self.h_std=args.h_stddev
        self.equalization=args.equalization
        #self.pilot=(self.zcsequence(1,self.num_sub)).repeat(args.N_pilot,1)
        self.pilot=(self.zcsequence(3,self.num_sub)).repeat(args.N_pilot,1)


    def zcsequence(self,u, seq_length, q=0):
        """
        Generate a Zadoff-Chu (ZC) sequence.
        ----------
        u : int
            Root index of the the ZC sequence: u>0.
        seq_length : int
            Length of the sequence to be generated. Usually a prime number:
            u<seq_length, greatest-common-denominator(u,seq_length)=1.
        q : int
            Cyclic shift of the sequence (default 0).
        Returns
        -------
        zcseq : 1D ndarray of complex floats
            ZC sequence generated.
        """
        for el in [u,seq_length,q]:
            if not float(el).is_integer():
                raise ValueError('{} is not an integer'.format(el))
        if u<=0:
            raise ValueError('u is not stricly positive')
        if u>=seq_length:
            raise ValueError('u is not stricly smaller than seq_length')
        if np.gcd(u,seq_length)!=1:
            raise ValueError('the greatest common denominator of u and seq_length is not 1')

        cf = seq_length%2
        n = np.arange(seq_length)
        zcseq = np.exp( -1j * np.pi * u * n * (n+cf+2.*q) / seq_length)
        zcseq=torch.from_numpy(zcseq)
        real_part=torch.real(zcseq).float()
        img_part=torch.imag(zcseq).float()
        zcseq=torch.complex(real_part,img_part)
        return zcseq

    def compute_h_broadcast(self,batch_size,subchannel_num):
        #h_stddev=torch.full((batch_size//64,1,subchannel_num),self.h_std/(np.sqrt(2))).float()
        h_stddev=torch.full((batch_size,subchannel_num),self.h_std/(np.sqrt(2))).float()
        h_mean=torch.zeros_like(h_stddev).float()
        h_real=Variable(torch.normal(mean=h_mean,std=h_stddev)).float()
        h_img=Variable(torch.normal(mean=h_mean,std=h_stddev)).float()
        #h=torch.complex(h_real,h_img).repeat(1,64,1).reshape(-1,subchannel_num)
        h=torch.complex(h_real,h_img)
        if self.sorted==1:
            h_norm=torch.abs(h)
            h_norm_sorted,indice=torch.sort(h_norm,dim=1)
            h_out=torch.gather(h,1,indice)
        else:
            h_out=h
        return h_out

    def power_normalize(self,feature):
        in_shape=feature.shape
        batch_size=in_shape[0]
        z_in=feature.reshape(batch_size,-1)
        #symbol_each_image=z_in.shape[1]
        #z_in_mean=z_in.mean(dim=1).unsqueeze(1).repeat(1,symbol_each_image)
        #z_in=z_in-z_in_mean
        sig_pwr=torch.square(torch.abs(z_in))
        ave_sig_pwr=sig_pwr.mean(dim=1).unsqueeze(dim=1)
        z_in_norm=z_in/(torch.sqrt(ave_sig_pwr))
        inputs_in_norm=z_in_norm.reshape(in_shape)
        return inputs_in_norm
            
    def transmit(self,feature,snr,h):
        feature_shape=feature.shape
        #prepare h: [batch,S,M)
        h_broadcast=h.unsqueeze(dim=1).repeat(1,feature_shape[1],1)
        #prepare W: [batch,S,M)
        noise_stddev=(torch.sqrt(10**(-snr/10))/torch.sqrt(torch.tensor(2))).reshape(-1,1,1)
        noise_stddev_board=noise_stddev.repeat(1,feature_shape[1],feature_shape[2]).float()
        mean=torch.zeros_like(noise_stddev_board).float()
        noise_real=Variable(torch.normal(mean=mean,std=noise_stddev_board).cuda()).float()
        noise_img=Variable(torch.normal(mean=mean,std=noise_stddev_board).cuda()).float()
        w=torch.complex(noise_real,noise_img)
        #compute channel_out
        channel_out=h_broadcast*feature+w
        return channel_out
    def LS_est(self,x,y):
        #x:(b,1,M) y:C(b,1,M)
        #h:(b,1,M)
        #H_est=Y/X,H_est_2=(X^HX)^-1XhY 
        #->same to next
        #y=y.squeeze().unsqueeze(2)
        #x=torch.diag_embed(x.squeeze())
        #h_est=torch.bmm(torch.inverse(x),y)
        return (y/x)

    def MMSE_est(self,x,y,snr):
        #est=R_h(X^hXR_h+var_wI)^-1X^hy
        #x:(b,1,M) y:C(b,1,M)
        #h_std:(b,1),snr:(b,1)
        #h_est:(b,1,M)

        input_shape=x.shape
        batch_size=input_shape[0]
        subchannel_num=input_shape[2]

        w_noise_varian_r=((10**(-snr/10))).repeat(1,subchannel_num).float().cuda()
        #w_noise_varian_r=torch.zeros_like(w_noise_varian_r)
        w_noise_varian_i=torch.zeros_like(w_noise_varian_r)
        w_noise_varian=torch.complex(w_noise_varian_r,w_noise_varian_i)

        h_stddev=torch.full((batch_size,subchannel_num),self.h_std*self.h_std).float().cuda()
        #R_h
        R_h_r=torch.diag_embed(h_stddev)
        R_h_i=torch.zeros_like(R_h_r)
        R_h=torch.complex(R_h_r,R_h_i)
        y=y.squeeze().unsqueeze(2)
        x=torch.diag_embed(x.squeeze())
        
        mid=torch.bmm(torch.bmm(self.conj_transpose(x),x),R_h)+torch.diag_embed(w_noise_varian)
        h_est=torch.bmm(torch.bmm(torch.bmm(R_h,torch.inverse(mid)),self.conj_transpose(x)),y)
        return h_est

    def conj_transpose(self,x):
        return torch.conj(x).permute(0,2,1)   

    def forward(self, x,input_snr):
        #parameter:
        batch_size=x.shape[0]
        #set pilot C^[batch,1,M]
        Y_p=self.pilot.repeat(batch_size,1,1)
        Y_p_norm=Y_p.cuda()
        x_head_ave=torch.zeros_like(x).cuda()    
        inside_loss_ave=torch.zeros(1).cuda()
        snr=np.full(batch_size,input_snr)
        channel_snr=torch.from_numpy(snr).float().cuda().view(-1,1).cuda()

        for transmit_times in range (5):
            #prepare H:
            #h:C[batch,M]; h_attention:R[batch,2M]
            h=self.compute_h_broadcast(batch_size,self.num_sub).cuda()
            #h=torch.ones_like(h)
            h_norm=torch.abs(h)
            #transmit pilot to estimate h
            #Y_p_norm_head=self.transmit(Y_p_norm,channel_snr,h)
            #h_est_ls=self.LS_est(Y_p_norm,Y_p_norm_head).squeeze()
            #dist_ls=torch.dist(h,h_est_ls)
            #h_est_mmse=self.MMSE_est(Y_p_norm,Y_p_norm_head,channel_snr).squeeze()
            #dist_mmse=torch.dist(h,h_est_mmse)
            #h_est_mmse_norm=torch.abs(h_est_mmse)

            #h_est=h_est_mmse
            h_est=h
            h_est_mmse_norm=torch.abs(h_est)
            encoder_attention=h_est_mmse_norm
            PA_input=encoder_attention.repeat(1,self.num_sym)
            #h_est=h
            #encode
            #encode
            if self.tran_know_flag==1:
                encoded_out = self.attention_encoder(x,encoder_attention)
            elif self.tran_know_flag==0:
                encoded_out = self.encoder(x)
            #encoded_out = self.encoder(x)
            encoded_shape=encoded_out.shape
            z=encoded_out.view(batch_size,-1)
            complex_list=torch.split(z,(z.shape[1]//2),dim=1)
            encoded_out_complex=torch.complex(complex_list[0],complex_list[1])
            Y=encoded_out_complex.view(batch_size,self.num_sym,-1)            
            ####Power constraint for each sending:
            Y_norm_1=self.power_normalize(Y)
            if self.PA_flag==1:
                #h_norm=torch.abs(h)
                #Y_norm_1=Y
                #located_weight=self.PA(PA_input).view(batch_size,self.num_sym,self.num_sub)
                located_weight=self.En_PA(PA_input).unsqueeze(dim=1).repeat(1,self.num_sym,1)
                y_located=Y_norm_1*located_weight#+Y_norm_1
                Y_norm=self.power_normalize(y_located)
            else:
                Y_norm=Y_norm_1
            Y_norm_head=self.transmit(Y_norm,channel_snr,h)

            ##Equalization:
            if self.equalization==1:
                ##Common method with compensation::
                h_broadcast_est=h_est.unsqueeze(dim=1).repeat(1,Y_norm_head.shape[1],1)
                compensation_factor=(torch.conj(h_broadcast_est)/(torch.square(torch.abs(h_broadcast_est))))
                Y_norm_head=compensation_factor*Y_norm_head
                #Y_norm_head=Y_norm_head.view(batch_size,-1)
                #Y_norm_head_real=torch.cat((torch.real(Y_norm_head),torch.imag(Y_norm_head)),dim=1).view(encoded_shape).float()
                #decoder_input=Y_norm_head_real
           
            if self.PA_flag==1:
                located_weight=self.De_PA(PA_input).unsqueeze(dim=1).repeat(1,self.num_sym,1)
                feature_after_PA_input=Y_norm_head*located_weight#+Y_norm_head
            else:
                feature_after_PA_input=Y_norm_head

            feature_after_PA_input=feature_after_PA_input.view(batch_size,-1)
            Y_norm_head_real=torch.cat((torch.real(feature_after_PA_input),torch.imag(feature_after_PA_input)),dim=1)
            
            decoder_input=Y_norm_head_real.view(encoded_shape).float()
            if self.tran_know_flag==1:
                x_head = self.attention_decoder(decoder_input,encoder_attention)
            elif self.tran_know_flag==0:
                x_head = self.decoder(decoder_input)  
                #x_head= self.decoder(decoder_input)

            Y_norm_reshape=Y_norm_1.view(batch_size,-1)
            Y_norm_reshape_real=torch.cat((torch.real(Y_norm_reshape),torch.imag(Y_norm_reshape)),dim=1)
            inside_loss=torch.nn.functional.mse_loss(Y_norm_reshape_real,Y_norm_head_real)
            inside_loss_ave=inside_loss_ave+inside_loss

            x_head_ave=x_head_ave+x_head

        x_head_ave=x_head_ave/5
        inside_loss_ave=inside_loss_ave/5

        return x_head_ave,inside_loss_ave

class Attention_all_JSCC(nn.Module):
    def __init__(self,args):
        super(Attention_all_JSCC, self).__init__()
        attention_size_encoder=64
        attention_size_decoder=64
        self.attention_encoder = Attention_Encoder(args,attention_size_encoder)
        self.encoder = Encoder(args)
        self.attention_decoder=Attention_Decoder(args,attention_size_decoder)
        self.gama=args.gama
        self.SNR_min=args.input_snr_min
        self.SNR_max=args.input_snr_max
        self.num_sub=args.M
        self.num_sym=args.S
        #pilot_path=args.pilot_path
        self.h_std=args.h_stddev
        self.equalization=args.equalization
        self.tran_know_flag=args.tran_know_flag
        
        # Generate the pilot signal
        '''
        if not os.path.exists(pilot_path):
            bits = torch.randint(2, (args.M,2))
            torch.save(bits,pilot_path)
            pilot = (2*bits-1).float()
        else:
            bits = torch.load(pilot_path)
            pilot = (2*bits-1).float()
            pilot_c=torch.complex(pilot[:,0],pilot[:,1])
        '''
        self.pilot=(self.zcsequence(3,self.num_sub)).repeat(args.N_pilot,1)
        #self.pilot = pilot_c.repeat(args.N_pilot,1)
    def zcsequence(self,u, seq_length, q=0):
        """
        Generate a Zadoff-Chu (ZC) sequence.
        ----------
        u : int
            Root index of the the ZC sequence: u>0.
        seq_length : int
            Length of the sequence to be generated. Usually a prime number:
            u<seq_length, greatest-common-denominator(u,seq_length)=1.
        q : int
            Cyclic shift of the sequence (default 0).
        Returns
        -------
        zcseq : 1D ndarray of complex floats
            ZC sequence generated.
        """
        for el in [u,seq_length,q]:
            if not float(el).is_integer():
                raise ValueError('{} is not an integer'.format(el))
        if u<=0:
            raise ValueError('u is not stricly positive')
        if u>=seq_length:
            raise ValueError('u is not stricly smaller than seq_length')
        if np.gcd(u,seq_length)!=1:
            raise ValueError('the greatest common denominator of u and seq_length is not 1')

        cf = seq_length%2
        n = np.arange(seq_length)
        zcseq = np.exp( -1j * np.pi * u * n * (n+cf+2.*q) / seq_length)
        zcseq=torch.from_numpy(zcseq)
        real_part=torch.real(zcseq).float()
        img_part=torch.imag(zcseq).float()
        zcseq=torch.complex(real_part,img_part)
        return zcseq

    def compute_h_fast_broadcast(self,batch_size,subchannel_num,h_num):
        h_stddev=torch.full((batch_size,subchannel_num),np.sqrt((self.h_std*self.h_std)/2)).float().unsqueeze(dim=1)
        h_mean=torch.zeros_like(h_stddev).float()
        h_out=torch.full((batch_size,h_num,subchannel_num),0).float()
        for i in range(h_num):
            h_real=Variable(torch.normal(mean=h_mean,std=h_stddev)).float()
            h_img=Variable(torch.normal(mean=h_mean,std=h_stddev)).float()
            h=torch.complex(h_real,h_img)
            if i ==0:
                h_out=h
            else:
                h_out=torch.cat((h_out,h),dim=1)
        return h_out
    def compute_h_slow_broadcast(self,batch_size,subchannel_num):
        h_stddev=torch.full((batch_size,subchannel_num),np.sqrt((self.h_std*self.h_std)/2)).float()
        h_mean=torch.zeros_like(h_stddev).float()
        h_real=Variable(torch.normal(mean=h_mean,std=h_stddev)).float()
        h_img=Variable(torch.normal(mean=h_mean,std=h_stddev)).float()
        h=torch.complex(h_real,h_img)
        return h

    def power_normalize(self,feature):
        in_shape=feature.shape
        batch_size=in_shape[0]
        z_in=feature.reshape(batch_size,-1)
        sig_pwr=torch.square(torch.abs(z_in))
        ave_sig_pwr=sig_pwr.mean(dim=1).unsqueeze(dim=1)
        z_in_norm=z_in/(torch.sqrt(ave_sig_pwr))
        inputs_in_norm=z_in_norm.reshape(in_shape)
        return inputs_in_norm

    def bmm_float_complex(self,float,complex):
        real=torch.real(complex)
        img=torch.imag(complex)
        real_out=torch.bmm(float,real)
        img_out=torch.bmm(float,img)
        return torch.complex(real_out,img_out)
    
    def transmit(self,feature,snr,h):
        feature_shape=feature.shape
        #prepare h: [batch,S,M)
        h_broadcast=h.unsqueeze(dim=1).repeat(1,feature_shape[1],1)
        
        #prepare W: [batch,S,M)
        noise_stddev=(torch.sqrt(10**(-snr/10))/torch.sqrt(torch.tensor(2))).reshape(-1,1,1)
        noise_stddev_board=noise_stddev.repeat(1,feature_shape[1],feature_shape[2]).float()
        mean=torch.zeros_like(noise_stddev_board).float()
        noise_real=Variable(torch.normal(mean=mean,std=noise_stddev_board).cuda()).float()
        noise_img=Variable(torch.normal(mean=mean,std=noise_stddev_board).cuda()).float()
        w=torch.complex(noise_real,noise_img)
        
        #compute channel_out
        channel_out=h_broadcast*feature+w
        #compensation_factor=(torch.conj(h_broadcast)/(torch.square(torch.abs(h_broadcast))))
        #channel_out_equal=compensation_factor*channel_out
        #channel_out_equal_2=feature+compensation_factor*w
        #torch.dist(channel_out_equal,channel_out_equal_2)
        return channel_out
    def LS_est(self,x,y):
        #x:(b,1,M) y:C(b,1,M)
        #h:(b,1,M)
        #H_est=Y/X,H_est_2=(X^HX)^-1XhY 
        return (y/x)

    def MMSE_est(self,x,y,snr):
        #est=R_h(X^hXR_h+var_wI)^-1X^hy
        #x:(b,1,M) y:C(b,1,M)
        #h_std:(b,1),snr:(b,1)
        #h_est:(b,1,M)
        input_shape=x.shape
        batch_size=input_shape[0]
        subchannel_num=input_shape[2]

        w_noise_varian_r=((10**(-snr/10))).repeat(1,subchannel_num).float().cuda()
        #w_noise_varian_r=torch.zeros_like(w_noise_varian_r)
        w_noise_varian_i=torch.zeros_like(w_noise_varian_r)
        w_noise_varian=torch.complex(w_noise_varian_r,w_noise_varian_i)

        h_stddev=torch.full((batch_size,subchannel_num),self.h_std*self.h_std).float().cuda()
        #R_h
        R_h_r=torch.diag_embed(h_stddev)
        R_h_i=torch.zeros_like(R_h_r)
        R_h=torch.complex(R_h_r,R_h_i)
        y=y.squeeze().unsqueeze(2)
        x=torch.diag_embed(x.squeeze())
        
        mid=torch.bmm(torch.bmm(self.conj_transpose(x),x),R_h)+torch.diag_embed(w_noise_varian)
        h_est=torch.bmm(torch.bmm(torch.bmm(R_h,torch.inverse(mid)),self.conj_transpose(x)),y)
        return h_est

    def conj_transpose(self,x):
        return torch.conj(x).permute(0,2,1)   

    def forward(self,x,input_snr):
        #parameter:
        batch_size=x.shape[0]
        #set pilot C^[batch,1,M]
        Y_p=self.pilot.repeat(batch_size,1,1)
        Y_p_norm=Y_p.cuda()
        #Y_p_norm=self.power_normalize(Y_p).cuda()
        #prepare snr: channel_snr,channel_snr_attention: R^[batch,1];
        x_head_ave=torch.zeros_like(x).cuda()
        if input_snr=='random':
            snr=np.random.rand(batch_size,)*(self.SNR_max-self.SNR_min)+self.SNR_min
        else:
            snr=np.full(batch_size,input_snr)
        channel_snr=torch.from_numpy(snr).float().cuda().view(-1,1).cuda()
        channel_snr_attention=channel_snr.unsqueeze(dim=2).repeat(1,self.num_sub,1)

        for transmit_times in range (10):
            #prepare H:
            #h:C[batch,M]; h_attention:R[batch,2M]
            h=self.compute_h_fast_broadcast(batch_size,self.num_sub).cuda()
            h_norm=torch.abs(h)
            
            encoder_attention=h_norm
            decoder_attention=h_norm

            #transmit pilot to estimate h
            Y_p_norm_head=self.transmit(Y_p_norm,channel_snr,h)
            h_est_ls=self.LS_est(Y_p_norm,Y_p_norm_head).squeeze()
            h_est_ls_norm=torch.abs(h_est_ls)
            dist_ls_norm=torch.dist(h_norm,h_est_ls_norm)
            h_est_mmse=self.MMSE_est(Y_p_norm,Y_p_norm_head,channel_snr).squeeze()
            h_est_mmse_norm=torch.abs(h_est_mmse)
            dist_mmse_norm=torch.dist(h_norm,h_est_mmse_norm)
            #h_attention=torch.cat((torch.real(h_est_mmse),torch.imag(h_est_mmse)),dim=1)
            #h_attention=torch.cat((torch.real(h_est_mmse),torch.imag(h_est_mmse)),dim=1)
            #h_attention_ls_norm=h_est_ls_norm
            #h_attention=torch.stack((torch.real(h_est_mmse),torch.imag(h_est_mmse)),dim=2)
            #h_attention=h_est_mmse_norm.unsqueeze(dim=2)
            h_est=h_est_mmse

            #encode
            if self.tran_know_flag==1:
                encoded_out = self.attention_encoder(x,encoder_attention)
            elif self.tran_know_flag==0:
                encoded_out = self.encoder(x)

            encoded_shape=encoded_out.shape
            #reshape and get feature to transmitter
            z=encoded_out.view(batch_size,-1)
            complex_list=z.view(batch_size,-1,2)
            real_list=complex_list[:,:,0]
            img_list=complex_list[:,:,1]
            encoded_out_complex=torch.complex(real_list,img_list)
            #complex_list=torch.split(z,(z.shape[1]//2),dim=1)
            #encoded_out_complex=torch.complex(complex_list[0],complex_list[1])
            Y=encoded_out_complex.view(batch_size,self.num_sym,-1)            
            ####Power constraint for each sending:
            Y_norm=self.power_normalize(Y)
            transmit_shape=Y_norm.shape
            Y_norm_head=self.transmit(Y_norm,channel_snr,h)
            ##Equalization:
            if self.equalization==1:
                ##Common method with compensation::
                h_broadcast_est=h_est.unsqueeze(dim=1).repeat(1,Y_norm_head.shape[1],1)
                compensation_factor=(torch.conj(h_broadcast_est)/(torch.square(torch.abs(h_broadcast_est))))
                Y_norm_head=compensation_factor*Y_norm_head
            elif self.equalization==2:
                #concat all
                h_broadcast=h.unsqueeze(dim=1).repeat(1,Y_norm_head.shape[1],1)
                compensation_factor=(torch.conj(h_broadcast)/(torch.square(torch.abs(h_broadcast))))
                Y_norm_head_com=compensation_factor*Y_norm_head
                #Y_norm_head=torch.cat((Y_norm_head,Y_norm_head_com),dim=1)

            Y_norm_head=Y_norm_head.view(batch_size,-1)
            Y_norm_head=torch.cat((torch.real(Y_norm_head),torch.imag(Y_norm_head)),dim=1)
            decoder_input=Y_norm_head.view(encoded_shape).float()

            #make the input for decoder:
            #Y_p_de=torch.cat((torch.real(Y_p),torch.imag(Y_p)),dim=2)
            #Y_p_head_de=torch.cat((torch.real(Y_p_head),torch.imag(Y_p_head)),dim=2)
            #decoder_input=torch.cat((Y_p_de,Y_p_head_de,Y_head_de),dim=1).view(batch_size,-1,encoded_shape[2],encoded_shape[3]).float()
            x_head= self.attention_decoder(decoder_input,decoder_attention)
            x_head_ave=x_head_ave+x_head

        x_head_ave=x_head_ave/10

        return x_head_ave
