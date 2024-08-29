from mmengine.registry import MODELS
from mmengine.model import BaseModule
from ..utils.layers import Embedding, ReduceDim, N2PAttention, GlobalDownSample, LocalDownSample, UpSample, furthest_point_sample, DTMEmbedding
from torch import nn
from einops import reduce, pack, repeat, rearrange
import torch
from ..utils.ops import select_knn


@MODELS.register_module()
class NewAPESSegBackbone(BaseModule):
    def __init__(self, which_ds, init_cfg=None):
        super(NewAPESSegBackbone, self).__init__(init_cfg)
        self.embedding = Embedding()
        self.dtm_embedding = DTMEmbedding()

        if which_ds == 'global':
            self.ds1 = GlobalDownSample(256)  # 512 pts -> 256 pts
            self.ds2 = GlobalDownSample(128)  # 256 pts -> 128 pts
        elif which_ds == 'local':
            self.ds1 = LocalDownSample(256)  # 512 pts -> 256 pts
            self.ds2 = LocalDownSample(128)  # 256 pts -> 128 pts
        else:
            raise NotImplementedError
        self.fps = furthest_point_sample
        self.n2p_attention0 = N2PAttention()
        self.n2p_attention1_1 = N2PAttention()
        self.n2p_attention1_2 = N2PAttention()
        self.n2p_attention2_1 = N2PAttention()
        self.n2p_attention2_2 = N2PAttention()
        self.n2p_attention3 = N2PAttention()
        self.n2p_attention4 = N2PAttention()
        self.ups1 = UpSample()
        self.ups2 = UpSample()
        # self.conv0 = nn.Sequential(nn.Conv1d(10000, 4096, 1, bias=False), nn.BatchNorm1d(4096), nn.LeakyReLU(0.2))
        # self.conv0 = nn.Sequential(nn.Conv1d(128, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        # self.conv0_1 = nn.Sequential(nn.Conv1d(128, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        self.conv1 = nn.Sequential(nn.Conv1d(128, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(16, 64, 1, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
        # self.conv3 = nn.Sequential(nn.Conv1d(12240, 2048, 1, bias=False), nn.BatchNorm1d(2048), nn.LeakyReLU(0.2)) 
        self.relu = nn.ReLU()

    def forward(self, x, shape_class):
        # ph = x[1]
        # ph = repeat(ph, 'B C 1 -> B C N', N=2048) # (B, 10000, 1) -> (B, 10000, 2048)
        # ph = self.conv0(ph)  # (B, 10000, 2048) -> (B, 4096, 2048)

        x_emb = self.embedding(x[0]).to(dtype=torch.float32)  # (B, 3, 2048) -> (B, 128, 2048)
        x_dtm = self.dtm_embedding(x[0]).to(dtype=torch.float32)
        # x_dtm = self.conv0(x_dtm).to(dtype=torch.float32)
        # x_emb = torch.cat([x_emb, x_dtm], dim=1) # 256
        # tmp = self.relu(self.conv0(x_emb))
        # x_emb = (self.conv0(x_emb) + tmp).to(dtype=torch.float32)
        x0 = self.n2p_attention0(x_emb)  # (B, 128, 2048) -> (B, 128, 2048)
        
        x1_fps_idx = self.fps(rearrange(x0, 'B C N -> B N C').contiguous(), 512).long()
        x1_fps = torch.gather(x0, 2, x1_fps_idx.unsqueeze(1).expand(-1, 128, -1))  # (B, 128, 2048) -> (B, 128, 512)
        x1_n2p = self.n2p_attention1_1(x1_fps) # x1_n2p
        x1_ds_idx = self.ds1(x1_n2p, True) # (B, 128, 512) -> (B, 128, 256)
        x1_ds_idx = torch.gather(x1_fps_idx, 1, x1_ds_idx) # (B, 256)
        x1_ds = torch.gather(x0, 2, x1_ds_idx.unsqueeze(1).expand(-1, 128, -1))  # (B, 128, 2048) -> (B, 128, 256)
        x1_knn = select_knn(x0, x1_ds, x1_ds_idx, 8, 1024)  # (B, 128, 2048) -> (B, 128, 1024)
        x1 = self.n2p_attention1_2(x1_knn) # (B, 128, 1024) -> (B, 128, 1024)

        x2_fps_idx = self.fps(rearrange(x1, 'B C N -> B N C').contiguous(), 256).long()
        x2_fps = torch.gather(x1, 2, x2_fps_idx.unsqueeze(1).expand(-1, 128, -1))  # (B, 128, 1024) -> (B, 128, 256)
        x2_n2p = self.n2p_attention2_1(x2_fps)
        x2_ds_idx = self.ds2(x2_n2p, True) # (B, 128, 256) -> (B, 128, 128)
        x2_ds_idx = torch.gather(x2_fps_idx, 1, x2_ds_idx)  
        x2_ds = torch.gather(x1, 2, x2_ds_idx.unsqueeze(1).expand(-1, 128, -1))  # (B, 128, 1024) -> (B, 128, 128)
        x2_knn = select_knn(x1, x2_ds, x2_ds_idx, 8, 512)  # (B, 128, 1024) -> (B, 128, 512)
        x2 = self.n2p_attention2_2(x2_knn)  # (B, 128, 512) -> (B, 128, 512)

        tmp = self.ups2(x1, x2)  # (B, 128, 512) -> (B, 128, 1024)
        x1 = self.n2p_attention3(tmp)  # (B, 128, 1024) -> (B, 128, 1024)
        tmp = self.ups1(x0, x1)  # (B, 128, 1024) -> (B, 128, 2048)
        x0 = self.n2p_attention4(tmp)  # (B, 128, 2048) -> (B, 128, 2048)
        x = self.conv1(x0)  # (B, 128, 2048) -> (B, 1024, 2048)
        x_max = reduce(x, 'B C N -> B C', 'max')  # (B, 1024, 2048) -> (B, 1024)
        x_avg = reduce(x, 'B C N -> B C', 'mean')  # (B, 1024, 2048) -> (B, 1024)
        x, _ = pack([x_max, x_avg], 'B *')  # (B, 1024) -> (B, 2048)
        shape_class = self.conv2(shape_class)  # (B, 16, 1) -> (B, 64, 1)
        x, _ = pack([x, shape_class], 'B *')  # (B, 2048) -> (B, 2112)
        x = repeat(x, 'B C -> B C N', N=2048)  # (B, 2112) -> (B, 2112, 2048)
        x, _ = pack([x, x0], 'B * N')  # (B, 2112, 2048) -> (B, 2240, 2048)
        x, _ = pack([x, x_dtm], 'B * N')  # (B, 2240, 2048) -> (B, 7240, 2048) / (B, 3264, 2048) 6336 12240
        # x = self.conv3(x)
        return x
