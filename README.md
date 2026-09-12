# Katrilyonluk Fraktal Hiper-Geometrik Yapay Zeka Mimarisi (1000-Gen / Bingen Simetrisi)

Bu proje, geleneksel doğrusal ve karesel büyüyen derin öğrenme mimarilerine (Transformer, RNN vb.) alternatif olarak geliştirilmiş, **dördüncü boyut hiper-cisimlerinin (Tesseract / 120-Hücreli yapılar)** ve fraktal geometrinin katlanma dinamiklerini temel alan özgün bir yapay sinir ağı mimarisidir.

## 📐 Matematiksel Temeller ve Katman Örüntüsü

Model, her bir adımda 1000 kat büyüyen benzersiz bir geometrik matruşka (fraktal yayılım) algoritmasına dayanır (n = 1000):

- **0. Katman (Merkez Çekirdek - 3B Bingen):** 1 adet 1000-gen (1000⁰). Toplam \(2n^2 - n = 1.999.000\) parametrelik iç hacme ve soyutlama kapasitesine sahiptir.
- **1. Katman (İç Küre Yüzeyi - 2B Bingen Bulutu):** Kesişmeyen 1000¹ = 1.000 adet 1000-gen. Toplam \(1.000.000\) algı köşesi içerir.
- **2. Katman (Dış Küre Yüzeyi - Yüksek Çözünürlüklü Algı Duvarı):** Kesişmeyen \(1000^2 = 1.000.000\) adet 1000-gen. Toplam \(1.000.000.000\) (1 Milyar) algı köşesi barındırır.

### 🌐 Katmanlar Arası Bağlantı Gücü
- **0-1 Katmanları Arası (İç Etkileşim):** n × n² = n³ → **1 Milyar sinaps (bağ)**.
- **1-2 Katmanları Arası (Dış Rezonans):** n² × n³ = n⁵ → **1 Katrilyon sinaps (bağ)**.

## 🚀 Donanım ve Bellek Sanallaştırması (Kronecker İllüzyonu)

Normal şartlar altında 1 Katrilyon bağı ve 1 Milyar köşeyi bilgisayar hafızasında doğrusal matrisler halinde tutmak **Terabaytlarca VRAM/RAM** gerektirir. 

Bu kısıtlamayı aşmak için modelde **Kronecker Çarpımı (Kronecker Product)** ve **Düşük Dereceli Matris Çarpanlarına Ayırma (Low-Rank Decomposition)** teknikleri uygulanmıştır. \(1.000.000 \times 1.000.000\) boyutundaki devasa işlem yükleri, 1000 × 1000 boyutunda küçük geometrik merceklere bölünmüştür. Bu sayede modelin RAM tüketimi **4 Gigabayttan sadece 8 Megabayta** düşürülerek standart donanımlarda kilitlenme yaşanmadan akıcı bir şekilde eğitilmesi sağlanmıştır.

## 🛠️ Klasör Yapısı

```text
hiper_geometrik_ai/
├── README.md               # Bu manifesto ve kullanım kılavuzu
├── gereksinimler.txt       # Bağımlılıklar (torch, numpy)
├── calistir.py             # Ana tetikleyici ve simülasyon motoru
├── mimari/                 # Geometrik katmanlar
│   ├── __init__.py
│   ├── encoder.py          # Veriyi 3B Bingen çekirdeğine haritalayan modül
│   ├── kuresel_bag.py      # Katrilyonluk optimize sanal bağlar
│   ├── decoder.py          # Radyal odaklama ve çıktı üretici
│   └── kuresel_model.py    # Bütünleşik AI gövdesi
└── egitim/                 # Öğrenme algoritmaları
    ├── __init__.py
    ├── kuresel_loss.py     # Geometrik bükülme hata fonksiyonu
    └── egitici.py          # Sıfır bellekli kararlı SGD eğitim motoru
```

## ⚙️ Çalıştırma

Gerekli bağımlılıkları yükledikten sonra:
```bash
pip install -r gereksinimler.txt
python calistir.py
```
