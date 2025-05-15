
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

#트레이닝 파라미터 설정
lr=1e-3
batch_size=4
num_epoch=100

data_dir='./datasets' #데이터가 저장되어있는 디렉토리
ckpt_dir='./checkpoint' #train된 network가 저장될 checkpoing 디렉토리
log_dir='./log' #텐서보드 log file이 저장될 log 디렉토리

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#unet network 구축하기
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
    #convolution + batch_normalization + ReLU
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers=[]
            layers+=[nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size,stride=stride,padding=padding,
                            bias=True)]
            layers+=[nn.BatchNorm2d(num_features=out_channels)]
            layers+=[nn.ReLU()]

            cbr=nn.Sequential(*layers)

            return cbr
        
        #contracting path
        self.enc1_1=CBR2d(in_channels=1,out_channels=64)
        self.enc1_2=CBR2d(in_channels=64,out_channels=64)

        self.pool1=nn.MaxPool2d(kernel_size=2)

        self.enc2_1=CBR2d(in_channels=64, out_channels=128)
        self.enc2_2=CBR2d(in_channels=128,out_channels=128)

        self.pool2=nn.MaxPool2d(kernel_size=2)

        self.enc3_1=CBR2d(in_channels=128, out_channels=256)
        self.enc3_2=CBR2d(in_channels=256,out_channels=256)

        self.pool3=nn.MaxPool2d(kernel_size=2)

        self.enc4_1=CBR2d(in_channels=256, out_channels=512)
        self.enc4_2=CBR2d(in_channels=512,out_channels=512)

        self.pool4=nn.MaxPool2d(kernel_size=2)

        self.enc5_1=CBR2d(in_channels=512,out_channels=1024)

        #Expansive path
        self.dec5_1=CBR2d(in_channels=1024,out_channels=512)

        self.unpool4=nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2=CBR2d(in_channels=2*512,out_channels=512) #in_channels가 두배인 이유는 encoder의 일부가 붙기때문(skip connection)
        self.dec4_1=CBR2d(in_channels=512,out_channels=256)
    
        self.unpool3=nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2=CBR2d(in_channels=2*256,out_channels=256) 
        self.dec3_1=CBR2d(in_channels=256,out_channels=128)

        self.unpool2=nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2=CBR2d(in_channels=2*128,out_channels=128) 
        self.dec2_1=CBR2d(in_channels=128,out_channels=64)

        self.unpool1=nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2=CBR2d(in_channels=2*64,out_channels=64) 
        self.dec1_1=CBR2d(in_channels=64,out_channels=64)
        
        self.fc=nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1,stride=1,padding=0,bias=True)

    def forward(self,x):
        enc1_1=self.enc1_1(x)
        enc1_2=self.enc1_2(enc1_1)
        pool1=self.pool1(enc1_2)

        enc2_1=self.enc2_1(pool1)
        enc2_2=self.enc2_2(enc2_1)
        pool2=self.pool2(enc2_2)

        enc3_1=self.enc3_1(pool2)
        enc3_2=self.enc3_2(enc3_1)
        pool3=self.pool3(enc3_2)

        enc4_1=self.enc4_1(pool3)
        enc4_2=self.enc4_2(enc4_1)
        pool4=self.pool4(enc4_2)

        enc5_1=self.enc5_1(pool4)

        dec5_1=self.dec5_1(enc5_1)

        unpool4=self.unpool4(dec5_1)
        cat4=torch.cat([unpool4, enc4_2], dim=1) #dim=[0:batch, 1:channel, 2:height, 3:width]
        dec4_2=self.dec4_2(cat4)
        dec4_1=self.dec4_1(dec4_2)

        unpool3=self.unpool3(dec4_1)
        cat3=torch.cat([unpool3, enc3_2],dim=1)
        dec3_2=self.dec3_2(cat3)
        dec3_1=self.dec3_1(dec3_2)

        unpool2=self.unpool2(dec3_1)
        cat2=torch.cat([unpool2,enc2_2], dim=1)
        dec2_2=self.dec2_2(cat2)
        dec2_1=self.dec2_1(dec2_2)

        unpool1=self.unpool1(dec2_1)
        cat1=torch.cat([unpool1, enc1_2], dim=1)
        dec1_2=self.dec1_2(cat1)
        dec1_1=self.dec1_1(dec1_2)

        x=self.fc(dec1_1)
        
        return x
    
#데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir=data_dir
        self.transform=transform

        lst_data=os.listdir(self.data_dir)

        lst_label=[f for f in lst_data if f.startswith('label')]
        lst_input=[f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label=lst_label
        self.lst_input=lst_input

    def __len__(self):
        return len(self.lst_label)
    
    def __getitem__(self, index):
        label=np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input=np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label=label/255.0
        input=input/255.0

        if label.ndim==2:
            label=label[:,:,np.newaxis]
        if input.ndim==2:
            input=input[:,:,np.newaxis]

        data={'input': input, 'label': label}

        if self.transform:
            data=self.transform(data)

        return data

#data transform 함수 구현하기
class ToTensor(object):
    def __call__(self, data):
        label, input=data['label'],data['input']
        
        #image의 numpy 차원= (Y,X,CH)
        #image의 tensor 차원= (CH,Y,X)
        label=label.transpose(2,0,1).astype(np.float32)
        input=input.transpose(2,0,1).astype(np.float32)

        data={'label':torch.from_numpy(label), 'input':torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5): #왜 (0,1)이라고 안하는거지
        self.mean=mean
        self.std=std

    def __call__(self,data):
        label, input=data['label'],data['input']

        input=(input-self.mean)/self.std #input은 0아니면 1이기 때문에 normalization 안 해도 된다.

        data={'label':label, 'input':input}

        return data
    
class RandomFlip(object):
    def __call__(self,data):
        label, input=data['label'],data['input']

        if np.random.rand()>0.5:
            label=np.fliplr(label)
            input=np.fliplr(input)

        if np.random.rand()>0.5:
            label=np.flipud(label)
            input=np.flipud(input)

        data={'label':label, 'input':input}

        return data
#네트워크 학습하기    
#transform 함수 여러개 묶기
transform=transforms.Compose([Normalization(mean=0.5,std=0.5),RandomFlip(),ToTensor()])

dataset_train=Dataset(data_dir=os.path.join(data_dir,'train'),transform=transform)
loader_train=DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

dataset_val=Dataset(data_dir=os.path.join(data_dir,'val'),transform=transform)
loader_val=DataLoader(dataset_val, batch_size=4, shuffle=False, num_workers=8) #일반적으로 train은 True, val과 test는 False

#네트워크 생성하기
net=UNet().to(device)

#손실함수 정의하기
fn_loss=nn.BCEWithLogitsLoss().to(device)

#optimizer 설정하기
optim=torch.optim.Adam(net.parameters(),lr=lr)

#그 밖에 부수적인 variables 설정하기
num_data_train=len(dataset_train)
num_data_val=len(dataset_val)

num_batch_train=np.ceil(num_data_train/batch_size)
num_batch_val=np.ceil(num_data_val/batch_size) #왜 4가아니고 batch_size냐

#그 밖에 부수적인 functions 설정하기
fn_tonumpy=lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1) #tensor->numpy
fn_denorm=lambda x, mean, std: (x+std)+mean #denormalization
fn_class=lambda x:1.0*(x>0.5)

#Tensorboard를 사용하기 위한 SummaryWriter 설정
writer_train=SummaryWriter(log_dir=os.path.join(log_dir,'train'))
writer_val=SummaryWriter(log_dir=os.path.join(log_dir,'val'))

#네트워크 저장하기
def save(ckpt_dir,net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(),'optim':optim.state_dict()},
               "./%s/model_epoch%d.pth"%(ckpt_dir,epoch))

#네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch=0
        return net, optim, epoch
    
    ckpt_lst=os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f:int(''.join(filter(str.isdigit,f))))

    dict_model=torch.load('./%s/%s'%(ckpt_dir,ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch=int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net,optim,epoch

#네트워크 학습시키기
st_epoch=0

net, optim,st_epoch=load(ckpt_dir=ckpt_dir, net=net, optim=optim)

for epoch in range(st_epoch+1,num_epoch+1):
    net.train() #dropout, batchnorm
    loss_arr=[]

    for batch, data in enumerate(loader_train,1):
        #forward pass
        label=data['label'].to(device)
        input=data['input'].to(device)

        output=net(input)

        #backward pass
        optim.zero_grad()
        
        loss=fn_loss(output,label)
        loss.backward()

        optim.step() #하이퍼파라미터 조정

        #손실함수 계산
        loss_arr+=[loss.item()]

        print('TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f' %
              (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

        #Tensorboard 저장하기
        label=fn_tonumpy(label)
        input=fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        output=fn_tonumpy(fn_class(output))

        writer_train.add_image('label',label, num_batch_train*(epoch-1)+batch, dataformats='NHWC') #n:배치, height, width, channel
        writer_train.add_image('input',input, num_batch_train*(epoch-1)+batch, dataformats='NHWC')
        writer_train.add_image('output',output,num_batch_train*(epoch-1)+batch, dataformats='NHWC')

    writer_train.add_scalar('loss', np.mean(loss_arr),epoch)

    #val은 backpropagation하는 부분이 없기 때문에 torch.no_grad()를 active
    with torch.no_grad():
        net.eval()
        loss_arr=[]

        for batch, data in enumerate(loader_val,1):
            #forward pass
            label=data['label'].to(device)
            input=data['input'].to(device)
            output=net(input)
            #(val은 backward pass 필요없음)

            #손실함수 계산하기
            loss_arr+=[loss.item()]

            print('VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f' %
                  (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))
            
            #Tensorboard 저장하기
            label=fn_tonumpy(label)
            input=fn_tonumpy(fn_denorm(input,mean=0.5, std=0.5))
            output=fn_tonumpy(fn_class(output))

            writer_val.add_image('label',label,num_batch_val*(epoch-1)+batch,dataformats='NHWC')
            writer_val.add_image('input',input,num_batch_val*(epoch-1)+batch,dataformats='NHWC')
            writer_val.add_image('output',output,num_batch_val*(epoch-1)+batch, dataformats='NHWC')

    writer_val.add_scalar('loss',np.mean(loss_arr),epoch)

    if epoch%5==0: #5번마다 저장
        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch) 
writer_train.close()
writer_val.close()









