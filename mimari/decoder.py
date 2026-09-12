import torch
import torch.nn as nn

class FraktalDecoder(nn.Module):
    """
    1000-Gen Küresel Bağ Çıktısını Sözlük Skorlarına (Logits) Dönüştüren Katman
    """
    def __init__(self, n=1000, sozluk_boyutu=8000):
        super().__init__()
        self.anlam_cozucu = nn.Linear(n, sozluk_boyutu)

    def forward(self, x):
        return self.anlam_cozucu(x)

# Geriye dönük uyumluluk
AnlamCozucu = FraktalDecoder