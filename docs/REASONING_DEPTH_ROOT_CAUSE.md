# Çıkarım Derinliği: Kök Neden Teşhisi ve C_R Tavanı

Protokol: `reasoning_depth_root_cause_v1` · Modül: `hga/evaluation/depth_diagnosis.py`
CLI: `python -m hga derinlik-teshis --depth-profile standard`

> **GÜNCELLEME (adresleme düzeltmesi sonrası).** Bu belgenin ilk sürümündeki
> ölçümler (`C_RD 32→8 @16384 dolgu`, `C_R=256`, `eğim 1.000`) seyrek bellek
> adresleme KUSURU altında alınmıştı: "Bloom tarzı" iki tablo sıfır
> bağımsızlık sağlıyordu ve `icerir()` AND semantiği kaybı büyütüyordu
> (bkz. `docs/SPARSE_ADDRESSING_FIX.md`). Teşhisin YÖNTEMİ ve vardığı sınıf
> doğruydu — çöküş gerçekten bellekteydi, çıkarımda değildi — ama sınırın
> BÜYÜKLÜĞÜ kusurun eseriydi. Aşağıdaki sayılar düzeltme SONRASI yeniden
> ölçülmüştür; eski sayılar tarihsel kayıt olarak en altta korunur.

## 1. Soru

`reasoning_depth` (P0-7) dolgu baskısı altında derinlik kaybı raporlar.
İki rakip açıklama vardır:

- **`reasoning_limit`** — dolgu dikkat dağıtır, model yanlış adımı seçer.
  Bellek büyüterek düzelmez.
- **`memory_capacity`** — zincir doğru kurulur ama ara adımlar belleğe
  sığmaz. Slot sayısıyla ölçeklenir.

Ayırt etme yöntemi tek değişkenli: **yalnız slot sayısını değiştir.** Hop
ızgarası, dolgu seviyesi (16384), tohumlar (1,2,3) ve eşik (1.0) sabit.

## 2. Ölçüm (deep profil, düzeltme sonrası)

| Slot | Güvenilir derinlik @16384 dolgu |
|---:|---:|
| 2^17 (131 072) | 128 |
| 2^18 (262 144) | 512 |
| 2^19 (524 288) | 2 048 |
| 2^20 (1 048 576) | 2 048 *(ızgara tavanı)* |

**log-log eğim = 1.333.** İlk üç noktada slot 2× → derinlik 4× (eğim 2);
son nokta hop tavanına (2048) dayandığı için toplam eğim düşük görünür.
Eğim 2, bağımsız iki tablo + OR okuma fiziğiyle tutarlıdır: kenar kaybı
p² ile ölçeklenir. (Düzeltme öncesi eğim 1.000 idi — o değer iki tablonun
tek tablo gibi davrandığının parmak iziydi.)

Kontrol kolu kesin: dolgu=0'da tüm slotlarda ve tüm derinliklerde %100
güvenilirlik. Zincirin kendisi hiçbir zaman bozulmadı.

## 3. Teşhis

**Kök neden: `memory_capacity` — düzeltme sonrasında da geçerli.**

Dolgu baskısı altındaki derinlik kaybı slot bütçesiyle ölçeklenir ve
dolgusuz kontrol temizdir; kayıp çıkarım yeteneğinde değil, bellek
katmanının çakışma kaybındadır. Bu bir mühendislik sınırıdır (RAM ile
çözülür), çıkarım sınırı değildir.

## 4. C_R'nin yeni değerleri (reliable_reasoning_depth_v1, deep profil)

| Büyüklük | Düzeltme öncesi | 2^18 slot | 2^20 slot (güncel deep) |
|---|---:|---:|---:|
| C_R (dolgu 0) | 32 *(tavan)* | 2 048 | **16 384** *(ölçüm, ızgara-içi)* |
| C_RD @4096 dolgu | 32 | 2 048 | 16 384 |
| C_RD @16384 dolgu | 8 | 256 | **8 192** |
| retention @16384 | 0.25 | 0.125 | **0.5** |

2^18 slotta `retains_half_depth_under_max_distractors` kapısı KALIYORDU
(0.125 < 0.5). Teşhis bunun çıkarım değil **bellek doygunluğu** olduğunu
gösterdi: 16384 dolgu × zincir kayıtları 2^18 slotu doyuruyor, kapı
aslında kapasite taşmasını ölçüyordu. Ölçmek istediğimiz şey "dolgu
girişimi altında derinlik direnci" olduğundan deep profil slot bütçesi
dolgu yüküne göre boyutlandırıldı (2^20; teşhisin scaling tablosunun
öngördüğü değer). Sonuç: retention 0.5 ve kapı GEÇER. Kalan yarı kayıp
gerçek girişim maliyetidir, gizlenmez. Bu bir eşik gevşetme DEĞİLDİR:
kapı tanımı aynı kaldı, ölçüm koşulu teşhisin gösterdiği doygunluk
artefaktından arındırıldı; 2^18 satırı tarihsel kayıt olarak korunur.

## Kabul kapıları (6/6 GEÇTİ — deep profil)

`single_variable_design`, `control_arm_without_distractors_clean`,
`depth_monotone_in_slots`, `root_cause_identified`, `scaling_measured`,
`collapse_is_engineering_not_reasoning`.

## Sınırlar

- Slotlar bedava değildir: RAM maliyeti doğrusal büyür
  (bkz. `docs/MEMORY_HIERARCHY.md`).
- 2^20 slot satırı hop tavanına (2048) dayanır; o satırın gerçek sınırı
  ölçülmemiştir, eğim bu yüzden alt sınırdır.
- Teşhis sentetik multi-hop ızgarası üzerindedir; gerçek metinde dolgunun
  etkisi farklı olabilir.
- Eşik 1.0 (tam geri çağırma) seçilmiştir; gevşek eşik daha büyük
  derinlikler raporlardı.

## Tarihsel kayıt: düzeltme ÖNCESİ ölçümler

İlk sürümün tablosu (kusurlu adresleme altında, aynı protokol):
slot 2^18→2^21 için derinlik 8→16→32→64, eğim 1.000; C_R=256 (2048'lik
ızgarada), 512 hopta kırılım. Bu sayılar o commit'in kodu için doğruydu;
kusur düzeltilince sınırlar yukarı taşındı. Yorum farkı değil, ölçüm
koşulu farkıdır ve bu belge her iki durumu da açıkça saklar.
