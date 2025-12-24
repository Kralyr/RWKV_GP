import numpy as np
import torch
import torch.nn as nn
import argparse
from Model.RWKV import RWKV


def conv_computer(input,conv_ks,conv_s):
    out=np.floor(
        ((input-conv_ks)/conv_s)+1
    )
    return out

class RWKV_GP(nn.Module):
    def __init__(self,config,snp_len,envday):
        super().__init__()

        self.dropout=config.dropout
        self.D_n_embd=config.D_n_embd

        self.dropout =config.dropout
        self.L1 = config.Linear1

        conv1_ks=8;conv1_s=4
        conv2_ks=4;conv2_s=2
        # pool1_ks=int(gene_length/1000);pool1_s=int(gene_length/1000)
        pool1_ks=8;pool1_s=8
        
        conv3_ks=4;conv3_s=4
        conv4_ks=4;conv4_s=4
        # pool2_ks=int(pool1_ks/2);pool2_s=int(pool1_ks/2)
        pool2_ks=4;pool2_s=4

        out=conv_computer(snp_len,conv1_ks,conv1_s)
        out=conv_computer(out,conv2_ks,conv2_s)
        out=conv_computer(out,pool1_ks,pool1_s)
        
        out=conv_computer(out,conv3_ks,conv3_s)
        out=conv_computer(out,conv4_ks,conv4_s)
        out=conv_computer(out,pool2_ks,pool2_s)
        
        self.n_channels=int(out)
        # print("n_channels",self.n_channels)

        # 这个是snp和env拼接之前降到多少维。目前是想和envday保持一致
        self.snp_feLen=64
        # envday就是env数据的第二维
        self.envday=envday

        # 第一层是5000直接占满显存
        # batch变大，这里3000变2000
        self.cnnSnp = nn.Sequential(
            
            nn.Conv1d(1, 64, conv1_ks, conv1_s),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 128, conv2_ks, conv2_s),
            nn.BatchNorm1d(128),

            nn.MaxPool1d(pool1_ks, pool1_s),
            nn.Dropout(0.5),

            nn.Conv1d(128, 128, conv3_ks, conv3_s),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 64, conv4_ks, conv4_s),
            nn.BatchNorm1d(64),

            nn.MaxPool1d(pool2_ks, pool2_s),
            nn.Dropout(self.dropout),


        )

        self.fcEnvFe= nn.Sequential(
            nn.Linear(self.envday, self.snp_feLen),
        )

        self.rwkv1 = RWKV(config,self.n_channels)
        self.rwkv2 = RWKV(config,16)
        self.rwkv3 = RWKV(config,17)

        self.jiang=1024

        self.fcjiang= nn.Sequential(
            nn.Linear( self.snp_feLen*self.D_n_embd,  self.jiang),
            nn.Dropout(self.dropout),
        )

        self.fc = nn.Sequential(
            nn.Linear( self.jiang*2, self.L1),
            nn.Dropout(self.dropout),
        )
        self.finalfc = nn.Sequential(
            nn.Linear(3*self.L1, self.L1),
            nn.Dropout(self.dropout),
            nn.Linear(self.L1, 1),
        )

    def forward(self, x1,x2,x3):
        # print("first",flush=True)
        x1=x1.unsqueeze(1)
        # print("x1",x1.shape)

        x1=self.cnnSnp(x1)
        # print("sx1",x1.shape)
        x1 = self.rwkv1(x1)
        # print("rx1",x1.shape)
        x1=x1.reshape(-1,1,self.snp_feLen*self.D_n_embd)

        # print(x2.shape)
        x2 = self.rwkv2(x2)
        # 下面这三行是为了将150降成64
        x2=x2.permute(0,2,1)
        x2=self.fcEnvFe(x2)
        x2=x2.permute(0,2,1)
        x2=x2.reshape(-1,1,self.snp_feLen*self.D_n_embd)

        x3 = self.rwkv3(x3)
        x3=x3.permute(0,2,1)
        x3=self.fcEnvFe(x3)
        x3=x3.permute(0,2,1)
        x3=x3.reshape(-1,1,self.snp_feLen*self.D_n_embd)

        x1=self.fcjiang(x1)
        x2=self.fcjiang(x2)
        x3=self.fcjiang(x3)

        x12=torch.cat((x1, x2), 2)
        x23=torch.cat((x2, x3), 2)
        x31=torch.cat((x3, x1), 2)
        
        x12=self.fc(x12)
        x23=self.fc(x23)
        x31=self.fc(x31)

        x=torch.cat((x12, x23, x31), 2)

        x=self.finalfc(x)
        # print(x.shape,flush=True)
        
        x=x.squeeze(1)
        # print(x.shape,flush=True)
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="RWKV_GP Model")
    parser.add_argument("--ctx_len", type=int, default=256, help="Context length (default: 256)")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (default: 0.5)")
    parser.add_argument("--Linear1", type=int, default=64, help="Linear1 dimension (default: 64)")

    parser.add_argument("--D_n_layer", type=int, default=8, help="Number of layers (default: 8)")
    parser.add_argument("--D_n_head", type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument("--D_n_embd", type=int, default=512, help="Embedding dimension (default: 512)")
    parser.add_argument("--D_n_attn", type=int, default=512, help="Attention dimension (default: 512)")
    parser.add_argument("--D_n_ffn", type=int, default=512, help="Feed-forward network dimension (default: 512)")
    parser.add_argument("--D_rwkv_emb_scale", type=float, default=0.4, help="RWKV embedding scale (default: 0.4)")
    parser.add_argument("--snp_len", type=int, default=5000, help="snp_len")

    return parser.parse_args()

if __name__ == "__main__":
    arg = parse_args()
    
    model = RWKV_GP(arg,snp_len=152864,envday=150)
    
    # 测试输入
    test_snp = torch.randn(39, 152864)
    envData = torch.randn(39, 150, 16)
    otherData = torch.randn(39, 150, 17)

    output = model(test_snp,envData,otherData)
    print(f"Output shape: {output.shape}")