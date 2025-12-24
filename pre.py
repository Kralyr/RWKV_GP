import torch
import pandas as pd
import numpy as np
from torch.utils import data
from scipy.stats import pearsonr
from Model.RWKV_GP import RWKV_GP
import os
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device,flush=True)


class myDataset(data.Dataset):
    def __init__(self, snp, envData,otherData,yy):
        self.snp = torch.as_tensor(snp)
        self.envData = torch.as_tensor(envData)
        self.otherData = torch.as_tensor(otherData)
        self.yy = torch.as_tensor(yy.iloc[:,1])
        self.Ma = yy.iloc[:,0]

    def __getitem__(self, index):
        return self.snp[index], self.envData,self.otherData,self.yy[index],self.Ma[index]

    def __len__(self):
        return len(self.snp)

sunEnvlist=['ALLSKY_SFC_SW_DWN','CLRSKY_SFC_SW_DWN','TOA_SW_DWN','ALLSKY_SFC_PAR_TOT','CLRSKY_SFC_PAR_TOT','ALLSKY_SFC_UVA',
            'ALLSKY_SFC_UVB','ALLSKY_SFC_UV_INDEX','T2M','T2MDEW','T2MWET','TS','T2M_RANGE','T2M_MAX','T2M_MIN','ALLSKY_SFC_LW_DWN']

otherEnvlist=['PS','WS2M','WS2M_MAX','WS2M_MIN','WS2M_RANGE','WD2M','WS10M','WS10M_MAX','WS10M_MIN','WS10M_RANGE','WD10M',
              'GWETTOP','GWETROOT','GWETPROF','QV2M','RH2M','PRECTOTCORR']


snpALL=pd.read_csv("./data507/maize976_15w_trans.txt")


for version in ['MO17','ZHENG58']:
    for env in ["CQ2012"]:#"DHN2011","HN2011","HN2012","YN2011"

        for sample in ['EarWeight']:
            dataLoad_path=f'{env}_{sample}_{version}'
            yy=pd.read_csv(f"./data507/YYfinal488/{sample}/{env}{sample}_{version}.csv")
            yy=yy.dropna()
            yy.index=list(range(len(yy)))

            useCol=list(yy['sample'])

            snp=snpALL[["ID"]+useCol]
            snp=snp.set_index('ID')
            snp=snp.T   
            
            ALLenvData=pd.read_csv(f'./envData/{env}.csv',skiprows=41,usecols=range(35)[2:])

            envData=ALLenvData[sunEnvlist]
            otherData=ALLenvData[otherEnvlist]

            envTensor=torch.tensor(envData.values,dtype=torch.float)
            otherTensor=torch.tensor(otherData.values,dtype=torch.float)
            
           
            trainsnp=snp.reset_index(drop=True)
            trainsnpTensor=torch.tensor(trainsnp.values, dtype=torch.float)
            
            trainset = myDataset(trainsnpTensor, envTensor,otherTensor,yy)
            
            
            train_loader = data.DataLoader(
                dataset=trainset,
                batch_size=8
            )
            

            save_path=f'./result_Pre_RWKV_GP/{version}/{sample}/{dataLoad_path}_result'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
         
            
            best_per=-1
            
            besti=0
            
            for i in range(1,2):
                
                best_test=pd.read_csv(f'./result_507_RWKV_GP/{sample}/{env}_{sample}_result/best_test{i}.csv')
                
                X=list(best_test['pred'])
                Y=list(best_test['exp'])

                pc = pearsonr(X,Y)

                if pc[0]>best_per:
                    best_per=pc[0]
                    besti=i
            
            
         
            pthfile =  f'./result_507_RWKV_GP/{sample}/{env}_{sample}_result/bestnet{besti}.pkl'

            # net = torch.load(pthfile, map_location='cpu')
            net = torch.load(pthfile, map_location=device)

            net.eval()
            
        
            pred_epoch = []
            exp_epoch = []

            Material = []
            
            for i, (tsnp, envData,otherData,tyy,tMa) in enumerate(train_loader, 0):

                tsnp = tsnp.to(device)
                tyy = tyy.to(device)
                envData = envData.to(device)
                otherData = otherData.to(device)

                pred = net(tsnp,envData,otherData)
                
                pred_epoch.extend(list(pred.squeeze(1).cpu().detach().numpy()))
                exp_epoch.extend(list(tyy.cpu().detach().numpy()))

                Material.extend(list(tMa))
            
            fin_pred =pd.DataFrame({'Material':Material,'exp':exp_epoch,'pred':pred_epoch})
            
            pc = pearsonr(exp_epoch,pred_epoch)
            
            print(dataLoad_path,pc[0],flush=True)
            
            # print('********************************************************')
            # print('********************************************************')
            # print('********************************************************')
            
            
            # exit()
            
            



