import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

class NgramDataset(Dataset):
    def __init__(self, token_ids, baglam_penceresi):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.baglam_penceresi = baglam_penceresi

    def __len__(self):
        return max(0, len(self.token_ids) - self.baglam_penceresi)

    def __getitem__(self, idx):
        x = self.token_ids[idx : idx + self.baglam_penceresi]
        y = self.token_ids[idx + self.baglam_penceresi]
        return x, y

class KureselEgitimMotoru:
    def __init__(self, model, ogrenme_hizi=0.002, toplam_cag=15):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=ogrenme_hizi, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=toplam_cag, eta_min=1e-5)

    def ngram_egitim_dongusu(self, kelimeler, tokenizer, cag_sayisi=15, batch_size=256):
        baglam = self.model.baglam_penceresi
        toplam_kelime = len(kelimeler)
        
        # 🚀 CPU LIMITI: Çağ başı 60.000 kelimelik kayan pencere
        DILIM_BOYUTU = min(60000, toplam_kelime)
        stride = max(1, (toplam_kelime - DILIM_BOYUTU) // max(1, cag_sayisi - 1)) if toplam_kelime > DILIM_BOYUTU else 0
        
        print(f"\n============================================================")
        print(f"   🚀 AKILLI SLIDING-WINDOW MOTORU (TÜM KORPUS TARANIYOR)")
        print(f"   📊 Toplam Korpus: {toplam_kelime:,} kelime | Çağ Dilimi: {DILIM_BOYUTU:,} | Batch: {batch_size}")
        print(f"============================================================")
        
        t_start = time.time()
        unk_id = getattr(tokenizer, "UNK_ID", getattr(tokenizer, "unk_id", 1))
        tum_ids = [tokenizer.kelime_to_id.get(w, unk_id) for w in kelimeler]
        print(f"  [✅ NDEKS] {toplam_kelime:,} kelime {(time.time()-t_start):.2f}s'de belleğe hazırlandı.\n")

        for cag in range(cag_sayisi):
            self.model.train()
            
            # Korpusta kayan pencere (Sliding Window Index)
            baslangic_idx = (cag * stride) if stride > 0 else 0
            bitis_idx = min(toplam_kelime, baslangic_idx + DILIM_BOYUTU)
            cag_ids = tum_ids[baslangic_idx:bitis_idx]
            
            dataset = NgramDataset(cag_ids, baglam)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            
            toplam_loss = 0.0
            adim = 0
            cag_basla = time.time()
            
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                toplam_loss += loss.item()
                adim += 1

            self.scheduler.step()
            ortalama_loss = toplam_loss / max(1, adim)
            gecen_sure = time.time() - cag_basla
            
            ilerleme = int((cag + 1) / cag_sayisi * 20)
            bar = "█" * ilerleme + " " * (20 - ilerleme)
            
            print(f"  Çağ {cag+1:02d}/{cag_sayisi:02d} | Loss: {ortalama_loss:.4f} | Süre: {gecen_sure:.1f}s | Dilim: [{baslangic_idx:,}-{bitis_idx:,}] |{bar}|")

        print("============================================================")
        print(f"  [🎉 TAMAMLANDI] Akıllı Tüm Korpus Eğitimi Bitti!")
        print("============================================================\n")