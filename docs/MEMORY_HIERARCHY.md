# Hiyerarşik Bellek Benchmarkı (P0-8)

- Protokol: `hierarchical_memory_v1` v1 (imza `379a04bf0bb4`)
- Profil / tohum: `standard` / `1`
- Kapasiteler: hot `1,000` / warm `10,000` / cold `30,000` / archive sınırsız
- Dayanıklı yazma (WAL fsync): `True`

## Katman dağılımı ve yazma

| Katman | Kayıt |
|---|---:|
| hot | 1,000 |
| warm | 10,000 |
| cold | 30,000 |
| archive | 9,000 |

Yazma: **2,088 kayıt/s** (478.9 µs/kayıt, toplam 50,000 kayıt 23.95 s)

## Okuma gecikmesi (katman başına)

| Katman | n | p50 (µs) | p95 (µs) | p99 (µs) | max (µs) |
|---|---:|---:|---:|---:|---:|
| hot | 9 | 1.41 | 1.68 | 1.70 | 1.70 |
| warm | 398 | 3.23 | 4.93 | 9.10 | 85.77 |
| cold | 1227 | 29.73 | 69.06 | 96.29 | 4195.35 |
| archive | 366 | 7989.39 | 14601.59 | 16660.11 | 30349.33 |

## Recall ve bütünlük

- Sonda sayısı: `2000`
- Bulunan: `2000` → **recall 1.000000**
- Değer bozulması: `0`

## Tahliye politikası (LRU)

- Sıcak küme: `20` anahtar, her biri `5` kez okundu
- Baskı yazması: `500`
- Sıcak kümenin hot'ta kalma oranı: **1.0000**
- Tahliyeler: `{'hot': 51510, 'warm': 41107, 'cold': 9873, 'archive': 0}`
- Terfiler: `{'hot': 2010, 'warm': 0, 'cold': 0, 'archive': 0}`

## Kaynak kullanımı

| Kaynak | Bayt |
|---|---:|
| RSS büyümesi | 12,107,776 |
| Cold (SQLite) | 2,826,240 |
| Archive (gzip) | 50,302 |
| WAL | 2,868,390 |
| **Disk toplam** | **5,744,932** |

## Çökme kurtarma

- Çökme sonrası RAM'de kalan: `0`
- WAL kaydı: `50,500`
- Hot'a geri yüklenen: `10,627`
- Kurtarma sonrası recall: **1.000000** (200 sonda)

## Kontrol grubu — arşiv KAPALI

- Düşen kayıt: `9,000`
- Bu ölçekte düşme bekleniyor mu: `True`

> Arşiv kapalıyken düşen kayıt sayısı; arşiv açıkken bu sayı 0 olmalıdır. Kontrol grubu, kayıpsızlık iddiasının boş bir ifade olmadığını gösterir.

## Kabul kapıları

| Kapı | Sonuç |
|---|---|
| no_data_loss_with_archive | GEÇTİ |
| no_value_corruption | GEÇTİ |
| all_tiers_exercised | GEÇTİ |
| hot_faster_than_cold | GEÇTİ |
| crash_recovery_complete | GEÇTİ |
| lru_retains_hot_set | GEÇTİ |
| capacity_bounds_respected | GEÇTİ |
| drop_control_group_behaves_as_expected | GEÇTİ |

## Bulgular

- Profil `standard`: 50,000 kayıt yazıldı, kapasiteler hot/warm/cold = 1,000/10,000/30,000.
- Katman dağılımı: hot 1,000, warm 10,000, cold 30,000, archive 9,000 (1 segment).
- Yazma hızı 2,088 kayıt/s (478.9 µs/kayıt, durable=True).
- Okuma gecikmesi: hot p50=1.4µs (n=9), warm p50=3.2µs (n=398), cold p50=29.7µs (n=1227), archive p50=7989.4µs (n=366).
- Cold okuma hot'tan **21.1× yavaş**. Tek bir ortalama gecikme sayısı bu farkı gizler; katman ayrımı bu yüzden var.
- Recall 1.0000 — 2000 sondanın hepsi bulundu, değer bozulması yok. Tahliye veri kaybı değil, katman değişimidir.
- Çökme sonrası: RAM'de 0 kayıt kaldı, WAL'den 10,627 kayıt hot'a geri yüklendi, kurtarma sonrası recall 1.0000.
- Kontrol grubu (arşiv KAPALI): 9,000 kayıt düştü. Arşiv açıkken 0 düştü — kayıpsızlık mekanizması gerçekten çalışıyor, tanım gereği doğru değil.
- RSS büyümesi 12.1 MB; disk 5.7 MB (cold 2.8 MB + archive 0.1 MB + wal 2.9 MB).

## Sınırlar

- Tek süreç, tek iş parçacığı; eşzamanlı erişim ve kilitlenme davranışı ÖLÇÜLMEMİŞTİR.
- 'Çökme' süreç sonlandırma değil, RAM katmanlarının temizlenmesi ile taklit edilir; gerçek kill -9 sırasında sayfa önbelleği davranışı farklı olabilir.
- Disk ölçümleri dosya boyutlarıdır; dosya sistemi blok doldurmasını ve sıkıştırma oranını içermez.
- RSS tüm süreci ölçer; başka nesnelerin payı ayrıştırılamaz.
- Profil `standard` gerçek 10⁸ ölçeğinin altındadır; kapasite iddiası ekstrapolasyon değil, ölçülen aralıkla sınırlıdır.
