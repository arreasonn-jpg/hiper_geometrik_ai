# Priority(E) Ağırlıkları: Elle Seçim Savunulamıyor

Protokol: `priority_weight_optimization_v1` · Modül: `hga/evaluation/priority_optimization.py`
CLI: `python -m hga oncelik-optimizasyon --signature-profile standard`

## Sorun

20 tohumlu ablasyon (`docs/SEED_STATISTICS_FINDINGS.md`) şunu bulmuştu:

> YÖN UYARISI: `w_gain` ve `w_conflict_penalty` terimlerini **KAPATMAK**
> doğrulama verimini **ARTIRIYOR** — bu havuzda katkıları negatif.

Bu, "ağırlıklar etkili mi?" sorusundan daha rahatsız edici bir soru
doğuruyordu: **ağırlıklar doğru mu?** Varsayılan değerler
(`w_gain=0.40`, `w_novelty=0.35`, `w_uncertainty=0.25`,
`w_conflict_penalty=0.20`) elle seçilmişti ve hiçbir veriye dayanmıyordu.

## Yöntem

Ağırlık uzayı (4 terim × 4 değer = 255 geçerli kombinasyon) **yalnız 10
eğitim tohumunda** tarandı. Bulunan en iyi set, aramada **hiç görülmemiş**
20 held-out tohumda test edildi. Izgara `0.0`'ı içerir — arama bir terimi
tamamen kapatabilmelidir, çünkü ablasyon bunun iyi olabileceğini
göstermişti.

Tohum sızıntısı kod seviyesinde yasaklıdır (`ValueError`), çünkü sızıntı
olsaydı ölçülen kazanç sahte olurdu.

## Sonuç

| Terim | Varsayılan (elle) | Aramanın bulduğu |
|---|---:|---:|
| `w_gain` | 0.40 | **0.00** |
| `w_novelty` | 0.35 | **0.00** |
| `w_uncertainty` | 0.25 | **0.00** |
| `w_conflict_penalty` | 0.20 | **0.20** |

| Küme | Varsayılan | Optimize | Kazanç |
|---|---:|---:|---:|
| Eğitim (10 tohum) | 0.3200 | 0.6300 | +0.3100 |
| **Held-out (20 tohum)** | 0.3300 | **0.6700** | **+0.3400** |

Held-out istatistiği: fark %95 CI `[0.265, 0.420]`, Hedges g = **+1.74**
(large), permütasyon p = **0.00005**. Kazanç ayrışıyor.

Genelleşme oranı **1.10** — kazanç held-out'ta tamamen korundu, hatta
biraz arttı. Aşırı uydurma yok.

## Yorum

Bu bir "ayar iyileştirmesi" değil, **modelin yanlış olduğunun kanıtı**.
Dört terimli ağırlıklı toplamın üç terimi bu havuzda yalnız gereksiz
değil, **zararlıydı**: hepsini sıfırlayıp tek başına çelişki cezasını
kullanmak doğrulama verimini iki katına çıkarıyor (0.33 → 0.67).

Priority(E) formülünün kendisi bu görev için fazla karmaşık. `InfoGain`,
`Novelty` ve `Uncertainty` terimleri aritmetik doğrulama havuzunda
sıralamayı gürültüyle bozuyor; tek yararlı sinyal çelişki cezası.

**Varsayılan ağırlıklar bu bulgu incelenmeden savunulamaz.** Modülü
varsayılanları değiştirmek için değil, bu iddiayı ölçülebilir kılmak için
yazdım; ağırlık değişikliği ayrı bir karardır ve gerçek alanlarda
tekrarlanmadan yapılmamalıdır.

## Ölçüm artefaktı uyarısı (ve düzeltmesi)

İlk koşumda duyarlılık tablosu `w_conflict_penalty`'yi "ölü ağırlık" diye
işaretliyordu. Bu yanlıştı: bu terim **tek aktif terim** olduğunda ölçeği
değiştirmek sıralamayı değiştirmez (monoton dönüşüm), bu yüzden eğri düz
görünür. Belirleyici olan terimin **varlığıdır**, ölçeği değil.

Rapor artık `scale_invariant` bayrağı taşıyor ve bu durumda terimi ölü
saymıyor. Test (`test_duyarlilik_olcum_artefakti_uretmiyor`) bu davranışı
zorunlu kılıyor.

## Kabul kapıları (9/9 GEÇTİ, standard profilde)

`train_test_seed_isolation`, `search_space_includes_zero`,
`holdout_evaluated`, `optimized_at_least_matches_default_on_holdout`,
`gain_survives_holdout`, `gain_statistically_distinguishable`,
`overfitting_measured`, `sensitivity_measured`, `paired_design`.

`smoke` profilinde (3 held-out tohum) anlamlılık kapısı **kasıtlı olarak
KALIR** — n=3'te ulaşılabilir en küçük p 0.25'tir.

## Sınırlar

- Izgara kabadır; bulunan nokta "ızgaranın en iyisi"dir, **küresel optimum
  değildir**.
- Hedef tek metriktir (doğrulama verimi). Çeşitlilik veya kapsama hedefi
  farklı ağırlıklar seçtirirdi.
- Havuz aritmetiktir, doğrulayıcı kapalı formdur; gerçek alanlarda optimal
  ağırlıklar farklı olabilir.
- Held-out tohumlar aynı havuz üreticisinden gelir: bu bir **tohum
  genellemesidir, alan genellemesi değildir.**
