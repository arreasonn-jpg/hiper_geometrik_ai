# Çıkarım Derinliği: Kök Neden Teşhisi ve C_R Tavanı

Protokol: `reasoning_depth_root_cause_v1` · Modül: `hga/evaluation/depth_diagnosis.py`
CLI: `python -m hga derinlik-teshis --depth-profile standard`

Karnedeki en düşük bölüm `reasoning` (6.0) idi. İki kapı kalıyordu:
`c_r_not_grid_limited` ve `retains_half_depth_under_max_distractors`.
Bu belge ikisinin de sebebini ölçer — ve biri için eski yorumu düzeltir.

## 1. Yanlış yorumlanmış bulgu

P0-7 şu tabloyu üretmişti:

| dolgu | 64 | 256 | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|---:|
| C_RD | 32 | 32 | 32 | 32 | **8** |

Bu, "dolgu baskısı altında çıkarım derinliği kaybı" diye raporlanmıştı.
**Bu yorum yanlıştı.** Ölçüm doğruydu, sebebi yanlış okunmuştu.

## 2. Ayırt edici deney

İki rakip açıklama vardı:

- **`reasoning_limit`** — dolgu dikkat dağıtır, model yanlış adımı seçer.
  Bellek büyüterek düzelmez.
- **`memory_capacity`** — zincir doğru kurulur ama ara adımlar belleğe
  sığmaz. Slot sayısıyla ölçeklenir.

Ayırt etme yöntemi tek değişkenli: **yalnız slot sayısını değiştir.** Hop
ızgarası, dolgu seviyesi (16384), tohumlar ve eşik (1.0) sabit.

| Slot | Güvenilir derinlik |
|---:|---:|
| 2^18 (262 144) | 8 |
| 2^19 (524 288) | 16 |
| 2^20 (1 048 576) | 32 |
| 2^21 (2 097 152) | 64 |

**Slot iki katına çıkınca derinlik iki katına çıkıyor. log-log eğim = 1.000.**

Kontrol kolu kesin: dolgu=0'da **tüm** derinlikler %100 güvenilir. Zincirin
kendisi hiçbir zaman bozulmadı.

## 3. Teşhis

**Kök neden: `memory_capacity`.**

C_RD tablosundaki 32→8 düşüşü bir yetenek eksikliği değil, bir
**yapılandırma seçimidir**. 16384 dolgu, 2^18 slotluk belleği doldurup
gerçek kayıtları çakışmaya zorluyor. Slot bütçesi verildiğinde derinlik
geri geliyor.

Bu önemli bir ayrımdır: mühendislik sınırı parayla/RAM'le çözülür, çıkarım
sınırı çözülmez. Eski yorum ikincisini iddia ediyordu, ölçüm birincisini
gösteriyor.

## 4. C_R'nin gerçek tavanı

`c_r_not_grid_limited` kapısı, C_R'nin ızgaranın en büyük değerine eşit
çıkması yüzünden kalıyordu — yani gerçek tavan bilinmiyordu. Izgara
genişletildi:

| Test edilen hop tavanı | C_R | Izgara sınırlı mı? |
|---:|---:|---|
| 32 | 32 | evet |
| 128 | 128 | evet |
| 2048 | **256** | **hayır** |

**C_R = 256.** 512 hop'ta güvenilirlik kırılıyor. Bu, önceki raporlanan
değerin (32) sekiz katıdır ve artık ızgara tavanı değil, gerçek bir
ölçümdür.

## Kabul kapıları (6/6 GEÇTİ)

`single_variable_design`, `control_arm_without_distractors_clean`,
`depth_monotone_in_slots`, `root_cause_identified`, `scaling_measured`,
`collapse_is_engineering_not_reasoning`.

`smoke` profilinde bellek sınırı görünmez ve teşhis `reasoning_limit`
döner; son kapı orada kasıtlı olarak KALIR. Teşhis bulamadığı şeyi
uydurmaz.

## Sınırlar

- Slotlar bedava değildir: RAM maliyeti doğrusal büyür
  (bkz. `docs/MEMORY_HIERARCHY.md`).
- **Gerçek çıkarım tavanına hâlâ ULAŞILMADI.** Bu ızgarada her seferinde
  bellek sınırına çarpıldı; HGA'nın gerçek çıkarım sınırı bu yüzden hâlâ
  ölçülmemiştir. Bilinen tek şey, ölçülen çöküşlerin sebebinin çıkarım
  olmadığıdır.
- Teşhis sentetik multi-hop ızgarası üzerindedir; gerçek metinde dolgunun
  etkisi farklı olabilir.
- Eşik 1.0 (tam geri çağırma) seçilmiştir; gevşek eşik daha büyük
  derinlikler raporlardı.
