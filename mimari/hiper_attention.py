import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HiperGeometrikAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        
        assert self.head_dim * num_heads == emb_dim, "emb_dim, num_heads'e tam bölünmelidir!"
        
        # 1. OPTİMİZASYON: Fused QKV Projeksiyonu (3 ayrı katman yerine tek büyük matris)
        self.qkv_proj = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        
        # 2. OPTİMİZASYON: Pre-LN Düzeni (Eğitim kararlılığı için katman öncesi normalizasyon)
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 3. OPTİMİZASYON: ReZero Yaklaşımı (Gradyanı kesmeyen öğrenilebilir skaler başlangıç)
        # out_proj ağırlıklarını tamamen sıfırlamak yerine bu parametreyi 0 ile başlatıyoruz.
        self.alpha = nn.Parameter(torch.zeros(1))
        
        self._reset_parameters()

    def _reset_parameters(self):
        # QKV ve Çıkış projeksiyonları için standart Xavier/Glorot başlatması
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Sequence_Length, Embedding_Dim)
        B, S, D = x.shape
        residual = x
        
        # 4. OPTİMİZASYON: Pre-LN Uygulaması
        x_norm = self.norm(x)
        
        # Fused QKV hesaplaması -> (B, S, 3*D)
        qkv = self.qkv_proj(x_norm)
        
        # Q, K, V parçalarına ayırma (Chunking) -> Her biri (B, S, D)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Çoklu kafa (Multi-head) için tensörleri yeniden şekillendirme
        # Son boyut düzeni Flash Attention için: (B, num_heads, S, head_dim) olmalıdır
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 5. OPTİMİZASYON: PyTorch 2.0+ Scaled Dot Product (Flash Attention) Entegrasyonu
        # Bu fonksiyon arkada donanım destekliyorsa bellek harcamayan Flash Attention çağırır.
        # Manuel matris çarpımı ve softmax adımlarını ortadan kaldırır.
        context = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False # Eğer decoder/GPT mimarisiyse True yapın
        )
        
        # Kafaları birleştirme (Merge heads) -> (B, S, D)
        context = context.transpose(1, 2).contiguous().view(B, S, D)
        
        # Çıkış projeksiyonu
        out = self.out_proj(context)
        
        # ReZero mekanizması ile çıktı ekleme: Başlangıçta alpha=0 olduğu için katman 
        # kimlik fonksiyonu (identity) gibi davranır ama alpha üzerinden alta gradyan akar!
        return residual + self.alpha * out
