#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch


# In[2]:


'''
FM fft矩阵[4096,4096]
IFM ifft矩阵[1024,1024]
'''

F=np.fft.fft(np.diag(np.ones([256]))).astype(np.complex64)
FM=np.zeros([16*256,16*256],dtype=np.complex64)
for i in range(16):
    FM[i*256:i*256+256,i*256:i*256+256]=F
    
IF=np.fft.ifft(np.diag(np.ones([256]))).astype(np.complex64)
IFM=np.zeros([4*256,4*256],dtype=np.complex64)
for i in range(4):
    IFM[i*256:i*256+256,i*256:i*256+256]=IF


# In[3]:


def make_M_from_csi_route(csi_route):
    '''
    求路径的卷积操作对应的矩阵
    '''
    M=np.zeros([256+32,256],dtype=np.complex64)
    pad=np.zeros(256+32,dtype=np.complex64)
    for i in range(256):
        e=32+i+1
        s=max(e-126,0)
        _len=e-s
        M[s:e,i]=csi_route[:_len][::-1]
    M[-32:]=M[-32:]+M[:32]
    M=M[32:]
    return M


# In[4]:


def make_csi_H(csi):
    '''
    求csi对应的矩阵(将调制解调过程:x->ifft->convolve->fft 用矩阵表示)
    csi的形状为[64,126]
    FM矩阵对应fft变换
    IFM矩阵对应ifft变换
    convolve_M矩阵对应convolve过程
    '''
    convolve_M=np.stack([make_M_from_csi_route(csi_route)for csi_route in csi],0)
    convolve_M=convolve_M.reshape([16,4,256,256]).transpose([1,2,0,3]).reshape([4*256,16*256])
    H=np.matmul(IFM,convolve_M)
    H=np.matmul(H,FM)
    return H


# In[5]:


def mino(x,csi):
    '''
    无噪音的mimo
    输出的长度为4096,可以reshape为[256,16,2]
    '''
    x=x.reshape([1024,2])*2-1
    x=0.7071*x[:,0]+0.7071j*x[:,1]
    M=make_csi_H(csi)
    y=np.matmul(x,M)
    y=y.reshape([16,256]).T
    y=np.stack([y.real,y.imag],-1).reshape([-1])
    return y


# In[6]:


def ofdm_simulate(data,csi,SNRdb):
    #data:[bs,4,256]
    data=torch.fft.ifft(data)
    data=conv(data,csi,SNRdb)
    data=torch.fft.fft(data)#[b,16,256]
    return data
def conv(data,csi,SNRdb):
    batch_size=data.shape[0]
    inputs=data.reshape([-1,1,1,256])
    pad=torch.zeros_like(inputs)[:,:,:,-126+1+32:]
    inputs = torch.cat([pad,inputs[:,:,:,-32:],inputs],-1)
    oup_real = torch.nn.functional.unfold(inputs.real,[1,126]).reshape([batch_size,4,126,256])
    oup_imag = torch.nn.functional.unfold(inputs.imag,[1,126]).reshape([batch_size,4,126,256])
    oup = oup_real+oup_imag*1j
    csi = csi.reshape([batch_size,16,4,126]).flip(-1)
    oup = torch.einsum('bwdc,bmwd->bmc',oup,csi)
    sigma2 = 0.0015 * 10 ** (-SNRdb / 10)
    noise = np.sqrt(sigma2 / 2) * torch.randn_like(oup)*2
    return oup+noise
class MIMO(torch.nn.Module):
    def __init__(self):
        super(MIMO,self).__init__()
        pilot='1001110011111101100111000110100000100101101010010010100110111100'
        pilot=np.asarray(list(pilot),dtype=np.float32)
        pilot = np.loadtxt('Pilot_32', delimiter=',').astype(np.float32)
        pilot = pilot.reshape([8,4,2])
        pilot = 0.7071 * (2 * pilot[:,:,0]- 1) + 0.7071j * (2 * pilot[:,:,1] - 1) #[8,4]
        pilot_zeros=np.zeros([256,4],dtype=np.complex64)
        for i in range(4):
            pilot_zeros[i*8::32,i]=pilot[:,i]
        pilot=pilot_zeros.reshape([256,4]).T
        self.pilot=torch.from_numpy(pilot).cuda()
    def forward(self,x,csi,SNRdb=10):
        batch_size=x.shape[0]
        x=x.reshape([batch_size,4,256,2])
        x=0.7071 * (2 * x[:,:,:,0]- 1) + 0.7071j * (2 * x[:,:,:,1] - 1)
        pilot=self.pilot.unsqueeze(0).expand([batch_size,-1,-1])
        A=ofdm_simulate(pilot,csi,SNRdb)
        B=ofdm_simulate(x,csi,SNRdb)#[b,16,256]
        oup=torch.stack([A,B],-1)#[b,16,256,2]
        oup=torch.stack([oup.real,oup.imag],-1)#[b,16,256,2,2]
        oup=oup.transpose(1,2)
        return oup#[b,256,16,2,2]


# In[7]:


if __name__=='__main__':
    csi=np.random.randn(64*126).astype(np.complex64).reshape([64,126])#形状为[64,126]的csi复矩阵
    x=np.random.randint(0,2,[2048]).astype(np.float32)
    y=mino(x,csi)
    mimo1=MIMO().cuda()
    y1=mimo1(torch.from_numpy(np.expand_dims(x,0)).cuda(),torch.from_numpy(np.expand_dims(csi,0)).cuda(),1000)[0,:,:,1]
    y1=y1.detach().cpu().numpy()
    print(np.abs(y-y1.reshape(-1)).max())
    print(np.abs(y-y1.reshape(-1)).max()/np.abs(y1).mean())#由于精度问题,会存在误差
