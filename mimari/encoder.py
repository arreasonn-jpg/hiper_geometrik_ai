import torch
import torch.nn as nn

class GeometrikVeriEncoder(nn.Module):
    def __init__(self, n=1000):
        super(GeometrikVeriEncoder, self).__init__()
        self.n = n
        
        # 4 GB'LIK MATRİS YERİNE SADECE BİRKAÇ MEGABAYTLIK IKI KÜÇÜK GEOMETRİK PROJEKSİYON
        # 512 boyutlu ham veriyi önce 1000 boyutuna süzüp, oradan sanal matris genişlemesi yapacağız.
        self.proj_A = nn.Linear(512, n, bias=False)
        self.proj_B = nn.Linear(n, n, bias=False)
        
    def forward(self, ham_veri_vektoru):
        batch_size = ham_veri_vektoru.size(0)
        
        # Veriyi iki farklı 1000 boyutlu geometrik izdüşüme ayır
        ara_A = self.proj_A(ham_veri_vektoru)
        ara_B = self.proj_B(ara_A)
        
        # 4 GB'lık 1.999.000 boyutunu RAM'de fiziksel olarak kurmak yerine,
        # matris çarpımı mantığıyla 1.000.000 boyutlu sanal bir alan (1000x1000) elde ediyoruz
        # ve bunu modelimizin çekirdek kapasitesine (n) benzeterek süzüyoruz.
        geometrik_izdusum = torch.matmul(ara_A.unsqueeze(2), ara_B.unsqueeze(1)).view(batch_size, -1)
        
        # Çekirdek kapasitesine sığması için (1.999.000 yerine 1.000.000 sanal köşe olarak) büküyoruz
        kuresel_veri = torch.tanh(geometrik_izdusum) 
        
        return kuresel_veri
