# İnsan Değerlendirme Dağıtım Paketi

Bu dizin `human_evaluation_protocol_v1` için **gerçek model yanıtlarıyla
doldurulmuş, kör** değerlendirme paketlerini içerir.

## İçerik

| Yol | Ne | Kime |
|---|---|---|
| `YONERGE.md` | Puanlama yönergesi (5 boyut, ölçekler) | her değerlendiriciye |
| `paketler/Rxx.json` | Değerlendiricinin göreceği 200 öğe (prompt + yanıt) | yalnız o değerlendiriciye |
| `paketler/Rxx_puanlama.csv` | Doldurulacak puan şablonu | yalnız o değerlendiriciye |
| `_GIZLI_degerlendiriciye_verme/` | Kör açma anahtarı | **kimseye verilmez** |
| `manifest.json` | Üretim imzaları ve sayımlar | arşiv |
| `google_forms/` | Google Forms'a bölünmüş aktarım + CSV doğrulama/dönüştürme | araştırma ekibi |

## Dağıtım adımları

1. Her değerlendiriciye ÜÇ dosya gönderin: `YONERGE.md`, kendi `Rxx.json`
   dosyası ve kendi `Rxx_puanlama.csv` dosyası. Farklı kişilere farklı
   `Rxx` verin; aynı paketi iki kişiye vermeyin.
2. `_GIZLI_degerlendiriciye_verme/` dizinini PAYLAŞMAYIN. İçindeki anahtar
   hangi yanıtın hangi sistemden geldiğini söyler; paylaşılırsa körleme
   bozulur ve sonuç geçersizleşir.
3. Değerlendirici, JSON'daki her öğeyi okuyup CSV'de aynı `item_id`
   satırına 5 puan yazar (4 × 1-5 ölçek + 1 × 0/1). Tüm 200 satır
   doldurulmalıdır.
4. Doldurulan CSV'leri `paketler/` altına aynı adla geri koyun
   (`R01_puanlama.csv` gibi). Google Forms kullanılacaksa CSV şablonu yerine
   `google_forms/KILAVUZ.md` içindeki aktarım/dönüştürme akışını izleyin;
   dönüştürücü aynı `R01_puanlama.csv` şemasını üretir.

## Analiz (puanlar toplandıktan sonra)

```python
from pathlib import Path
from hga.evaluation.human_eval_fill import collect_ratings_from_csv
from hga.evaluation.human_evaluation import build_human_evaluation_protocol

ratings, ozet = collect_ratings_from_csv(Path("insan_degerlendirme_paketleri"))
rapor = build_human_evaluation_protocol(collected_ratings=ratings)
# rapor.results → boyut başına Krippendorff α; kapılar otomatik güncellenir
```

En az 10 değerlendiricinin CSV'si dolmadan `human_ratings_collected`
kapısı açılmaz ve karnede bölüm dürüstçe `n/a` kalır.

## Dürüstlük notu

Neural kollar ~841K parametreli, 1.9M tokenlik korpusla eğitilmiş minik
dil modelleridir; Türkçeleri akıcı DEĞİLDİR ve bu beklenen durumdur.
Değerlendirme tam da bu kaliteyi ölçer — "modeller iyi yazar" iddiası
taşımaz. Dördüncü kol (kural tabanlı) bilmediğini açıkça söyleyen epistemik
katmandır; halüsinasyon boyutunda kontrast zeminidir.
