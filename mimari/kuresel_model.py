import torch
import torch.nn as nn
import torch.nn.functional as F
from kuresel_bag import OptimizeEdilmisKureselBag
from decoder import FraktalDecoder

class HiperGeometrikAI(nn.Module):
    def __init__(self, n=1000, baglam_penceresi=8, sozluk_boyutu=8000, emb_dim=64):
        super().__init__()
        self.n = n
        self.baglam_penceresi = baglam_penceresi
        
        self.kelime_gomme = nn.Embedding(sozluk_boyutu, emb_dim, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.randn(1, baglam_penceresi, emb_dim) * 0.02)
        
        # 🔥 MULTI-HEAD ATTENTION (GERÇEK DKKAT MEKANZMASI)
        self.mha = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.norm_attn = nn.LayerNorm(emb_dim)
        
        self.u_kure = nn.Linear(baglam_penceresi * emb_dim, n)
        self.v_kure = nn.Linear(baglam_penceresi * emb_dim, n)
        
        self.u_gate = nn.Linear(n, n)
        self.v_gate = nn.Linear(n, n)
        
        self.kuresel_bag = OptimizeEdilmisKureselBag(n)
        self.norm_bag = nn.LayerNorm(n)
        
        self.decoder = FraktalDecoder(n, sozluk_boyutu)

    def forward(self, x):
        seq_len = x.size(1)
        emb = self.kelime_gomme(x) + self.pos_encoder[:, :seq_len, :]
        
        # Multi-Head Attention
        attn_out, _ = self.mha(emb, emb, emb)
        emb = self.norm_attn(emb + attn_out)
        
        duz = emb.reshape(emb.size(0), -1)
        
        u = torch.relu(self.u_kure(duz))
        v = torch.relu(self.v_kure(duz))
        
        u = F.normalize(u, p=2, dim=-1)
        v = F.normalize(v, p=2, dim=-1)
        
        u_attn = u * torch.sigmoid(self.v_gate(v))
        v_attn = v * torch.sigmoid(self.u_gate(u))
        
        # Ana küresel bağ matrisi üretimi
        bag = self.kuresel_bag(u_attn, v_attn)
        
        # Tuple Dönüşüm Koruması (LayerNorm TypeError Çözümü)
        if isinstance(bag, tuple):
            bag = bag[0]
            
        bag = self.norm_bag(bag)
        logits = self.decoder(bag)
        return logits