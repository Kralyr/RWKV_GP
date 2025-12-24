import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import numpy as np
import pandas as pd
import datetime
import random
from scipy.stats import pearsonr
import copy
from sklearn.model_selection import KFold
from Model.RWKV_GP import RWKV_GP


dictMap={
    'EarWeight':{
        "CQ2012":{'D_ctx_len': 150, 'batch': 27, 'dropout': 0.39371534992464335, 'Linear1': 256, 'lr': 2.3352220123632906e-05},
        "DHN2011":{'D_ctx_len': 1024, 'batch': 16, 'dropout': 0.46763050258967204, 'Linear1': 256, 'lr': 0.0002456829259281119},
        "HN2011":{'D_ctx_len': 150, 'batch': 57, 'dropout': 0.3235735795193278, 'Linear1': 128, 'lr': 0.0001274186700054864},
        "HN2012":{'D_ctx_len': 256, 'batch': 20, 'dropout': 0.39316193566024576, 'Linear1': 256, 'lr': 0.0009054822910167285},
        "YN2011":{'D_ctx_len': 150, 'batch': 36, 'dropout': 0.24412638509753, 'Linear1': 128, 'lr': 0.00024599345523311635}
    }
}


def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():

    parser = argparse.ArgumentParser(description="RWKV Time Series Model")

    parser.add_argument("--model_id", type=str, default="RWKV_GP", help="model name")
    parser.add_argument("--D_n_layer", type=int, default=8, help="Number of layers (default: 8)")
    parser.add_argument("--D_n_head", type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument("--D_n_embd", type=int, default=512, help="Embedding dimension (default: 512)")
    parser.add_argument("--D_n_attn", type=int, default=512, help="Attention dimension (default: 512)")
    parser.add_argument("--D_n_ffn", type=int, default=512, help="Feed-forward network dimension (default: 512)")
    parser.add_argument("--Epoch", type=int, default=60, help="")
    parser.add_argument('--env', default='CQ2012', type=str)
    parser.add_argument('--phe', default='EarWeight', type=str)  #EarWeight  

    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training (e.g., 'cpu', 'cuda:0')")

    parser.add_argument("--ctx_len", type=int, default=150, help="Context length")
    parser.add_argument("--batch", type=int, default=27, help="Batch size")
    parser.add_argument("--dropout", type=float, default=0.39371534992464335, help="Dropout rate")
    parser.add_argument("--Linear1", type=int, default=256, help="Linear1 dimension)")
    parser.add_argument("--lr", type=float, default=2.3352220123632906e-05, help="Learning rate")
    
    return parser.parse_args()


class myDataset(data.Dataset):

    def __init__(self, snp, Data_Env, otherData, yy):
        self.snp = torch.as_tensor(snp)
        self.Data_Env = torch.as_tensor(Data_Env)
        self.otherData = torch.as_tensor(otherData)
        self.yy = torch.as_tensor(yy.iloc[:, 1])
        self.Ma = yy.iloc[:, 0]

    def __getitem__(self, index):
        return self.snp[index], self.Data_Env, self.otherData, self.yy[index], self.Ma[index]

    def __len__(self):
        return len(self.snp)


def main():
    args = parse_args()  
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    setup_seed(0)  

    snpALL = pd.read_csv("./Data_G/507like976_15w_trans.txt")
    print(snpALL)


    dataLoad_path = f'{args.env}_{args.phe}'
    print(dataLoad_path)

    yy = pd.read_csv(f"./Data_G/YYfinal/{args.phe}/{args.env}{args.phe}.csv")
    yy = yy.dropna()
    yy.index = list(range(len(yy)))
    print(yy, flush=True)

    useCol = list(yy['sample'])
    snp = snpALL[["ID"] + useCol]
    snp = snp.set_index('ID')
    snp = snp.T
    print(snp, flush=True)

    sunEnvlist=['ALLSKY_SFC_SW_DWN','CLRSKY_SFC_SW_DWN','TOA_SW_DWN','ALLSKY_SFC_PAR_TOT','CLRSKY_SFC_PAR_TOT','ALLSKY_SFC_UVA',
            'ALLSKY_SFC_UVB','ALLSKY_SFC_UV_INDEX','T2M','T2MDEW','T2MWET','TS','T2M_RANGE','T2M_MAX','T2M_MIN','ALLSKY_SFC_LW_DWN']

    otherEnvlist=['PS','WS2M','WS2M_MAX','WS2M_MIN','WS2M_RANGE','WD2M','WS10M','WS10M_MAX','WS10M_MIN','WS10M_RANGE','WD10M',
              'GWETTOP','GWETROOT','GWETPROF','QV2M','RH2M','PRECTOTCORR']

    ALL_Data_Env=pd.read_csv(f'./Data_Env/{args.env}.csv',skiprows=41,usecols=range(35)[2:])  #
    Data_Env=ALL_Data_Env[sunEnvlist]
    otherData=ALL_Data_Env[otherEnvlist]
    envTensor=torch.tensor(Data_Env.values,dtype=torch.float)
    otherTensor=torch.tensor(otherData.values,dtype=torch.float)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    perListzhe = []

    for kn, (train, test) in enumerate(kf.split(snp), 1):
        trainid = train.tolist()
        valid = test.tolist()

        trainsnp = snp.iloc[trainid].reset_index(drop=True)

        trainsnpTensor = torch.tensor(trainsnp.values, dtype=torch.float)
        tarinyy = yy.iloc[trainid].reset_index(drop=True)
        trainset = myDataset(trainsnpTensor, envTensor,otherTensor,tarinyy)

        valsnp = snp.iloc[valid].reset_index(drop=True)
        valsnpTensor = torch.tensor(valsnp.values, dtype=torch.float)
        valyy = yy.iloc[valid].reset_index(drop=True)
        valset = myDataset(valsnpTensor, envTensor,otherTensor,valyy)

        train_loader = data.DataLoader(
            dataset=trainset,
            batch_size=args.batch,
            num_workers=2,
            shuffle=True
        )
        val_loader = data.DataLoader(
            dataset=valset,
            batch_size=args.batch,
            num_workers=2,
            shuffle=False
        )

        
        net = RWKV_GP(args, snp_len=152864,envday=150).to(device)
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        loss_func = nn.SmoothL1Loss()

        train_loss_all = []
        val_loss_all = []

        train_pc_all = []
        test_pc_all = []

        train_pc_v_all = []
        test_pc_v_all = []

        fin_exp = []
        fin_pred = []
        fin_Material = []
        best_per = 0.0
        starttime = datetime.datetime.now()
        bestnet = copy.deepcopy(net)

        for epoch in range(args.Epoch):
            net.train()
            train_loss = 0.0
            train_pred_epoch = []
            train_exp_epoch = []

            for i, (tsnp, Data_Env, otherData, tyy, tMa) in enumerate(train_loader, 0):
                
                tsnp = tsnp.to(device)
                tyy = tyy.to(device)
                Data_Env = Data_Env.to(device)
                otherData = otherData.to(device)

                optimizer.zero_grad()
                pred = net(tsnp, Data_Env, otherData)
                tyy = tyy.unsqueeze(1)

                loss = loss_func(pred.float(), tyy.float())

                train_pred_epoch.extend(list(pred.squeeze(1).cpu().detach().numpy()))
                train_exp_epoch.extend(list(tyy.squeeze(1).cpu().detach().numpy()))

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            pc = pearsonr(train_pred_epoch, train_exp_epoch)
            train_pc_all.append(pc[0])
            train_pc_v_all.append(pc[1])
            train_loss_all.append(train_loss / len(train_loader))

            net.eval()
            val_loss = 0.0
            test_pred_epoch = []
            test_exp_epoch = []
            test_Material = []

            with torch.no_grad():
                for i, (tsnp, Data_Env, otherData, tyy, tMa) in enumerate(val_loader, 0):
                    tsnp = tsnp.to(device)
                    tyy = tyy.to(device)
                    Data_Env = Data_Env.to(device)
                    otherData = otherData.to(device)

                    pred =  net(tsnp, Data_Env, otherData)
                    tyy = tyy.unsqueeze(1)

                    test_pred_epoch.extend(list(pred.squeeze(1).cpu().detach().numpy()))
                    test_exp_epoch.extend(list(tyy.squeeze(1).cpu().detach().numpy()))
                    test_Material.extend(list(tMa))

                    val_loss += loss.item()

            pc = pearsonr(test_pred_epoch, test_exp_epoch)

            if pc[0] > best_per:
                best_per = pc[0]
                bestnet = copy.deepcopy(net)

                fin_exp = test_exp_epoch
                fin_pred = test_pred_epoch
                fin_Material = test_Material

            test_pc_all.append(pc[0])
            test_pc_v_all.append(pc[1])
            val_loss_all.append(val_loss / len(val_loader))


            endtime = datetime.datetime.now()
            consumed_time = (endtime - starttime).total_seconds()
            print(f"epoch:{epoch}  Consumed Time: {consumed_time:.1f} seconds", flush=True)
            starttime = endtime

        save_path = f'./result_507_{args.model_id}/{args.phe}/{dataLoad_path}_result'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        f = open(f"{save_path}/best_per.txt", 'a')
        f.write(f'best_per{i}' + f" {best_per}")
        if i==9:
            f.write('\n\n')
        else:
            f.write('\n')
        f.close()
        
        np.savetxt(save_path + f'/best_per{kn}.txt', [best_per])
        torch.save(bestnet, save_path + f'/bestnet{kn}.pkl')

        best_test = pd.DataFrame({'Material': fin_Material, 'exp': fin_exp, 'pred': fin_pred})
        best_test.to_csv(save_path+f'/best_test{kn}.csv',index=False)

        perListzhe.append(best_per)


    print(args.env,perListzhe,flush=True)

    print('data done')


if __name__ == '__main__':

    main()
