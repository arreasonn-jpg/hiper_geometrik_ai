# Çekirdek Tohum İstatistikleri (P1)

- Protokol: `core_seed_statistics_v1` v1 (imza `1c12f2b7c280`)
- Profil: `custom` · tohum sayısı: **20** (çekirdek kural: 20)

## Protokol tohum disiplini

| Protokol | Tohum | 20-tohum kuralı |
|---|---:|---|
| priority_ablation | 20 | GEÇTİ |
| signature | 20 | GEÇTİ |
| operator_baselines | 20 | GEÇTİ |

## İstatistiksel güç sınırı

- n = `20` eşleşmiş tohum
- Ulaşılabilir en küçük iki yönlü p: `0.000002`
- p<0.05 saptanabilir mi: **evet**
- p<0.01 saptanabilir mi: **evet**

> Eşleşmiş permütasyon testinde ulaşılabilecek en küçük iki yönlü p-değeri tohum sayısıyla sınırlıdır. Bu sınırın üstünde bir 'anlamlılık' iddia edilemez.

## Eşleşmiş karşılaştırmalar (76 adet, 41 tanesinde CI sıfırı dışlıyor)

| Bağlam | İşlem | Kontrol | Δ ortalama | Δ %95 CI | Cohen's d | perm p | Hüküm |
|---|---|---|---:|---|---:|---:|---|
| signature/A_long_chain | hga | symbolic | -0.5090 | [-0.5262, -0.4922] | -12.693 | 0.0000 | AYRIŞMA: p=0.00005, g=-12.1854 (large), CI [-0.526172, -0.492187] sıfırı içermiyor. |
| signature/A_long_chain | hga | dense | -0.0098 | [-0.0277, 0.0117] | -0.214 | 0.3859 | AYRIM YOK: fark CI'si [-0.0277342, 0.0117191] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/A_long_chain | hga | transformer | -0.0156 | [-0.0398, 0.0113] | -0.265 | 0.2546 | AYRIM YOK: fark CI'si [-0.0398437, 0.011328] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/A_long_chain | hga | hga_no_memory | +0.0000 | [0.0000, 0.0000] | +0.000 | 1.0000 | AYRIM YOK: fark CI'si [0, 0] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/A_long_chain | hga | hga_no_kronecker | -0.0012 | [-0.0242, 0.0219] | -0.022 | 0.9333 | AYRIM YOK: fark CI'si [-0.0242187, 0.021875] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/A_long_chain | hga | hga_no_attention | -0.0066 | [-0.0234, 0.0125] | -0.158 | 0.5059 | AYRIM YOK: fark CI'si [-0.0234374, 0.0125098] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/A_long_chain | hga | hybrid | -0.5090 | [-0.5262, -0.4922] | -12.693 | 0.0000 | AYRIŞMA: p=0.00005, g=-12.1854 (large), CI [-0.526172, -0.492187] sıfırı içermiyor. |
| signature/B_unseen_entity | hga | symbolic | -0.4949 | [-0.5121, -0.4773] | -12.103 | 0.0000 | AYRIŞMA: p=0.00005, g=-11.6188 (large), CI [-0.512109, -0.477344] sıfırı içermiyor. |
| signature/B_unseen_entity | hga | dense | +0.0238 | [0.0070, 0.0383] | +0.648 | 0.0119 | AYRIŞMA: p=0.01195, g=0.6221 (medium), CI [0.0070312, 0.0382813] sıfırı içermiyor. |
| signature/B_unseen_entity | hga | transformer | -0.0086 | [-0.0359, 0.0176] | -0.137 | 0.5505 | AYRIM YOK: fark CI'si [-0.0359374, 0.0175782] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/B_unseen_entity | hga | hga_no_memory | +0.0000 | [0.0000, 0.0000] | +0.000 | 1.0000 | AYRIM YOK: fark CI'si [0, 0] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/B_unseen_entity | hga | hga_no_kronecker | +0.0020 | [-0.0160, 0.0195] | +0.046 | 0.8041 | AYRIM YOK: fark CI'si [-0.0160155, 0.0195316] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/B_unseen_entity | hga | hga_no_attention | +0.0043 | [-0.0172, 0.0254] | +0.086 | 0.7137 | AYRIM YOK: fark CI'si [-0.0171975, 0.0253906] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/B_unseen_entity | hga | hybrid | -0.4949 | [-0.5121, -0.4773] | -12.103 | 0.0000 | AYRIŞMA: p=0.00005, g=-11.6188 (large), CI [-0.512109, -0.477344] sıfırı içermiyor. |
| signature/C_unseen_relation | hga | symbolic | -0.4941 | [-0.5105, -0.4777] | -12.914 | 0.0000 | AYRIŞMA: p=0.00005, g=-12.3978 (large), CI [-0.510547, -0.477735] sıfırı içermiyor. |
| signature/C_unseen_relation | hga | dense | +0.0035 | [-0.0207, 0.0281] | +0.061 | 0.7855 | AYRIM YOK: fark CI'si [-0.0207032, 0.028125] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/C_unseen_relation | hga | transformer | -0.0055 | [-0.0301, 0.0188] | -0.096 | 0.6790 | AYRIM YOK: fark CI'si [-0.030078, 0.0187599] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/C_unseen_relation | hga | hga_no_memory | +0.0000 | [0.0000, 0.0000] | +0.000 | 1.0000 | AYRIM YOK: fark CI'si [0, 0] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/C_unseen_relation | hga | hga_no_kronecker | +0.0113 | [-0.0055, 0.0281] | +0.285 | 0.2398 | AYRIM YOK: fark CI'si [-0.00546895, 0.0281348] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/C_unseen_relation | hga | hga_no_attention | +0.0219 | [0.0047, 0.0402] | +0.523 | 0.0314 | AYRIŞMA: p=0.03145, g=0.5018 (medium), CI [0.0046874, 0.0402342] sıfırı içermiyor. |
| signature/C_unseen_relation | hga | hybrid | -0.4941 | [-0.5105, -0.4777] | -12.914 | 0.0000 | AYRIŞMA: p=0.00005, g=-12.3978 (large), CI [-0.510547, -0.477735] sıfırı içermiyor. |
| signature/D_unseen_both | hga | symbolic | -0.5039 | [-0.5223, -0.4879] | -12.437 | 0.0000 | AYRIŞMA: p=0.00005, g=-11.9398 (large), CI [-0.522266, -0.487891] sıfırı içermiyor. |
| signature/D_unseen_both | hga | dense | +0.0141 | [-0.0035, 0.0324] | +0.335 | 0.1396 | AYRIM YOK: fark CI'si [-0.0035155, 0.0324219] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/D_unseen_both | hga | transformer | +0.0035 | [-0.0285, 0.0344] | +0.048 | 0.8236 | AYRIM YOK: fark CI'si [-0.0285156, 0.034375] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/D_unseen_both | hga | hga_no_memory | +0.0000 | [0.0000, 0.0000] | +0.000 | 1.0000 | AYRIM YOK: fark CI'si [0, 0] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/D_unseen_both | hga | hga_no_kronecker | +0.0039 | [-0.0141, 0.0219] | +0.092 | 0.7046 | AYRIM YOK: fark CI'si [-0.0140627, 0.0218749] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/D_unseen_both | hga | hga_no_attention | -0.0156 | [-0.0344, 0.0023] | -0.358 | 0.1199 | AYRIM YOK: fark CI'si [-0.0343751, 0.00234375] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/D_unseen_both | hga | hybrid | -0.5039 | [-0.5223, -0.4879] | -12.437 | 0.0000 | AYRIŞMA: p=0.00005, g=-11.9398 (large), CI [-0.522266, -0.487891] sıfırı içermiyor. |
| signature/E_conflict | hga | symbolic | -0.5059 | [-0.5234, -0.4883] | -12.263 | 0.0000 | AYRIŞMA: p=0.00005, g=-11.7721 (large), CI [-0.523437, -0.488281] sıfırı içermiyor. |
| signature/E_conflict | hga | dense | -0.0172 | [-0.0387, 0.0023] | -0.356 | 0.1306 | AYRIM YOK: fark CI'si [-0.0386719, 0.0023439] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/E_conflict | hga | transformer | -0.0051 | [-0.0250, 0.0156] | -0.106 | 0.6434 | AYRIM YOK: fark CI'si [-0.0250001, 0.0156251] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/E_conflict | hga | hga_no_memory | +0.0000 | [0.0000, 0.0000] | +0.000 | 1.0000 | AYRIM YOK: fark CI'si [0, 0] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/E_conflict | hga | hga_no_kronecker | +0.0055 | [-0.0121, 0.0262] | +0.124 | 0.5944 | AYRIM YOK: fark CI'si [-0.0121093, 0.0261718] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/E_conflict | hga | hga_no_attention | -0.0129 | [-0.0344, 0.0078] | -0.259 | 0.2788 | AYRIM YOK: fark CI'si [-0.0343748, 0.0078127] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/E_conflict | hga | hybrid | -0.2566 | [-0.2840, -0.2281] | -3.931 | 0.0000 | AYRIŞMA: p=0.00005, g=-3.7742 (large), CI [-0.283984, -0.228115] sıfırı içermiyor. |
| signature/F_memory_dependent | hga | symbolic | -0.0816 | [-0.1477, -0.0289] | -0.579 | 0.0004 | AYRIŞMA: p=0.00040, g=-0.5558 (medium), CI [-0.147656, -0.0288964] sıfırı içermiyor. |
| signature/F_memory_dependent | hga | dense | +0.3418 | [0.2754, 0.3969] | +2.418 | 0.0000 | AYRIŞMA: p=0.00005, g=2.3213 (large), CI [0.27539, 0.396875] sıfırı içermiyor. |
| signature/F_memory_dependent | hga | transformer | +0.1867 | [0.1098, 0.2617] | +1.059 | 0.0001 | AYRIŞMA: p=0.00015, g=1.0169 (large), CI [0.109766, 0.261719] sıfırı içermiyor. |
| signature/F_memory_dependent | hga | hga_no_memory | +0.4316 | [0.3586, 0.4918] | +2.780 | 0.0000 | AYRIŞMA: p=0.00005, g=2.6686 (large), CI [0.358584, 0.491797] sıfırı içermiyor. |
| signature/F_memory_dependent | hga | hga_no_kronecker | -0.0387 | [-0.0973, 0.0098] | -0.309 | 0.2013 | AYRIM YOK: fark CI'si [-0.0972656, 0.0097655] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/F_memory_dependent | hga | hga_no_attention | -0.0297 | [-0.1102, 0.0442] | -0.163 | 0.4748 | AYRIM YOK: fark CI'si [-0.110166, 0.0441505] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/F_memory_dependent | hga | hybrid | -0.0816 | [-0.1477, -0.0289] | -0.579 | 0.0004 | AYRIŞMA: p=0.00040, g=-0.5558 (medium), CI [-0.147656, -0.0288964] sıfırı içermiyor. |
| signature/G_epistemic | hga | symbolic | -0.6762 | [-0.6887, -0.6629] | -22.428 | 0.0000 | AYRIŞMA: p=0.00005, g=-21.5308 (large), CI [-0.688672, -0.662891] sıfırı içermiyor. |
| signature/G_epistemic | hga | dense | -0.0133 | [-0.0355, 0.0082] | -0.264 | 0.2474 | AYRIM YOK: fark CI'si [-0.0355469, 0.00820285] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/G_epistemic | hga | transformer | -0.0168 | [-0.0359, 0.0020] | -0.379 | 0.0998 | AYRIM YOK: fark CI'si [-0.0359375, 0.00195295] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/G_epistemic | hga | hga_no_memory | +0.0000 | [0.0000, 0.0000] | +0.000 | 1.0000 | AYRIM YOK: fark CI'si [0, 0] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/G_epistemic | hga | hga_no_kronecker | -0.0105 | [-0.0340, 0.0125] | -0.195 | 0.3982 | AYRIM YOK: fark CI'si [-0.0339843, 0.0125] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/G_epistemic | hga | hga_no_attention | +0.0016 | [-0.0219, 0.0250] | +0.029 | 0.9220 | AYRIM YOK: fark CI'si [-0.0218751, 0.0249999] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/G_epistemic | hga | hybrid | -0.4484 | [-0.4707, -0.4293] | -9.319 | 0.0000 | AYRIŞMA: p=0.00005, g=-8.9465 (large), CI [-0.470703, -0.429287] sıfırı içermiyor. |
| signature/H_distractor | hga | symbolic | -0.5023 | [-0.5129, -0.4910] | -19.518 | 0.0000 | AYRIŞMA: p=0.00005, g=-18.7376 (large), CI [-0.512891, -0.491016] sıfırı içermiyor. |
| signature/H_distractor | hga | dense | -0.0098 | [-0.0215, 0.0016] | -0.358 | 0.1213 | AYRIM YOK: fark CI'si [-0.0214846, 0.0015624] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/H_distractor | hga | transformer | -0.0352 | [-0.0492, -0.0219] | -1.109 | 0.0000 | AYRIŞMA: p=0.00005, g=-1.0644 (large), CI [-0.049219, -0.021875] sıfırı içermiyor. |
| signature/H_distractor | hga | hga_no_memory | +0.0000 | [0.0000, 0.0000] | +0.000 | 1.0000 | AYRIM YOK: fark CI'si [0, 0] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/H_distractor | hga | hga_no_kronecker | +0.0012 | [-0.0168, 0.0184] | +0.028 | 0.9335 | AYRIM YOK: fark CI'si [-0.0167971, 0.0183592] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/H_distractor | hga | hga_no_attention | +0.0098 | [-0.0055, 0.0258] | +0.266 | 0.2809 | AYRIM YOK: fark CI'si [-0.00546895, 0.0257811] sıfırı içeriyor; kollar ayrıştırılamadı. |
| signature/H_distractor | hga | hybrid | -0.5023 | [-0.5129, -0.4910] | -19.518 | 0.0000 | AYRIŞMA: p=0.00005, g=-18.7376 (large), CI [-0.512891, -0.491016] sıfırı içermiyor. |
| operator_baselines/kronecker_teacher | kronecker | rank1 | -0.8696 | [-0.8897, -0.8472] | -17.366 | 0.0000 | AYRIŞMA: p=0.00005, g=-16.6713 (large), CI [-0.889657, -0.847209] sıfırı içermiyor. |
| operator_baselines/kronecker_teacher | kronecker | low_rank_param_matched | -0.7559 | [-0.7825, -0.7263] | -11.399 | 0.0000 | AYRIŞMA: p=0.00005, g=-10.9429 (large), CI [-0.782473, -0.726336] sıfırı içermiyor. |
| operator_baselines/kronecker_teacher | kronecker | low_rank_flop_matched | -0.4009 | [-0.4266, -0.3752] | -6.580 | 0.0000 | AYRIŞMA: p=0.00005, g=-6.3168 (large), CI [-0.426613, -0.375237] sıfırı içermiyor. |
| operator_baselines/kronecker_teacher | kronecker | kron_sum_2 | -0.0010 | [-0.0012, -0.0008] | -2.166 | 0.0000 | AYRIŞMA: p=0.00005, g=-2.0797 (large), CI [-0.00122695, -0.000822498] sıfırı içermiyor. |
| … | | | | | | | (16 satır daha) |

## Priority(E) ağırlık ablasyonu — tohum dağılımı

| Ağırlık | Metrik | n | Ortalama | Std | %95 CI |
|---|---|---:|---:|---:|---|
| w_gain | kendall_tau | 20 | 0.5368 | 0.0702 | [0.5058, 0.5659] |
| w_gain | topk_overlap | 20 | 0.2350 | 0.1309 | [0.1800, 0.2900] |
| w_gain | downstream_delta::verification_yield | 20 | 0.2800 | 0.2375 | [0.1750, 0.3800] |
| w_novelty | kendall_tau | 20 | 0.9491 | 0.0156 | [0.9421, 0.9553] |
| w_novelty | topk_overlap | 20 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| w_novelty | downstream_delta::verification_yield | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| w_uncertainty | kendall_tau | 20 | 0.8921 | 0.0224 | [0.8827, 0.9016] |
| w_uncertainty | topk_overlap | 20 | 0.8350 | 0.1040 | [0.7900, 0.8800] |
| w_uncertainty | downstream_delta::verification_yield | 20 | -0.0200 | 0.0410 | [-0.0400, -0.0050] |
| w_conflict_penalty | kendall_tau | 20 | 0.9416 | 0.0179 | [0.9337, 0.9491] |
| w_conflict_penalty | topk_overlap | 20 | 0.8900 | 0.0852 | [0.8550, 0.9250] |
| w_conflict_penalty | downstream_delta::verification_yield | 20 | 0.0200 | 0.0616 | [-0.0050, 0.0450] |

## Kabul kapıları

| Kapı | Sonuç |
|---|---|
| all_core_protocols_use_same_seeds | GEÇTİ |
| all_core_protocols_meet_20_seeds | GEÇTİ |
| comparisons_are_paired | GEÇTİ |
| every_comparison_reports_ci | GEÇTİ |
| every_comparison_reports_effect_size | GEÇTİ |
| every_comparison_reports_two_tests | GEÇTİ |
| power_limit_documented | GEÇTİ |
| design_can_reach_p_0_05 | GEÇTİ |

## Bulgular

- 20 tohum × 3 çekirdek protokol; 76 eşleşmiş karşılaştırma üretildi.
- Çekirdek tohum kuralı (20) karşılandı; bu protokoller artık 'duman testi' değil bilimsel sonuçtur.
- Ulaşılabilir en küçük iki yönlü p = 0.000002. Tasarım p<0.05 saptayabilir.
- Karşılaştırmaların 41/76 tanesinde farkın %95 güven aralığı sıfırı DIŞLIYOR; geri kalanı istatistiksel olarak ayırt edilemez. Ayırt edilemeyen sonuçlar gizlenmedi.
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.00010173, 8.05651e-06] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0035155, 0.0324219] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.00546895, 0.0257811] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.00546895, 0.0281348] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0121093, 0.0261718] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0140627, 0.0218749] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0160155, 0.0195316] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0167971, 0.0183592] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0171975, 0.0253906] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0207032, 0.028125] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0214846, 0.0015624] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0218751, 0.0249999] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0234374, 0.0125098] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0242187, 0.021875] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0250001, 0.0156251] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0277342, 0.0117191] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0285156, 0.034375] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.030078, 0.0187599] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0339843, 0.0125] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0343748, 0.0078127] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0343751, 0.00234375] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0355469, 0.00820285] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0359374, 0.0175782] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0359375, 0.00195295] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0386719, 0.0023439] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0398437, 0.011328] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.0972656, 0.0097655] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [-0.110166, 0.0441505] sıfırı içeriyor; kollar ayrıştırılamadı.': 1 karşılaştırma
-   └ hüküm 'AYRIM YOK: fark CI'si [0, 0] sıfırı içeriyor; kollar ayrıştırılamadı.': 7 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=-1.0644 (large), CI [-0.049219, -0.021875] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=-10.9429 (large), CI [-0.782473, -0.726336] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=-11.6188 (large), CI [-0.512109, -0.477344] sıfırı içermiyor.': 2 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=-11.7721 (large), CI [-0.523437, -0.488281] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=-11.9398 (large), CI [-0.522266, -0.487891] sıfırı içermiyor.': 2 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=-12.1854 (large), CI [-0.526172, -0.492187] sıfırı içermiyor.': 2 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=-12.3978 (large), CI [-0.510547, -0.477735] sıfırı içermiyor.': 2 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=-16.6713 (large), CI [-0.889657, -0.847209] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=-18.7376 (large), CI [-0.512891, -0.491016] sıfırı içermiyor.': 2 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=-2.0797 (large), CI [-0.00122695, -0.000822498] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=-21.5308 (large), CI [-0.688672, -0.662891] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=-3.7742 (large), CI [-0.283984, -0.228115] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=-6.3168 (large), CI [-0.426613, -0.375237] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=-8.9465 (large), CI [-0.470703, -0.429287] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=13.4769 (large), CI [0.561679, 0.59698] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=17.4453 (large), CI [0.861993, 0.90368] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=17.4465 (large), CI [0.861912, 0.903587] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=17.4565 (large), CI [0.861854, 0.903533] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=17.6080 (large), CI [0.861308, 0.902547] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=194.0253 (large), CI [0.956445, 0.960522] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=2.3213 (large), CI [0.27539, 0.396875] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=2.6686 (large), CI [0.358584, 0.491797] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=23.2508 (large), CI [0.313485, 0.324899] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=4.2271 (large), CI [0.0445634, 0.054324] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=4.5279 (large), CI [0.0778167, 0.0932853] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=4.7894 (large), CI [0.262706, 0.311949] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=45.8944 (large), CI [0.932577, 0.94917] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=45.8962 (large), CI [0.932586, 0.949182] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=5.5023 (large), CI [0.0311549, 0.0361769] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00005, g=6.3170 (large), CI [0.0563543, 0.064053] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00015, g=1.0169 (large), CI [0.109766, 0.261719] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00040, g=-0.5558 (medium), CI [-0.147656, -0.0288964] sıfırı içermiyor.': 2 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.00110, g=0.8946 (large), CI [0.00366615, 0.0101841] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.01195, g=0.6221 (medium), CI [0.0070312, 0.0382813] sıfırı içermiyor.': 1 karşılaştırma
-   └ hüküm 'AYRIŞMA: p=0.03145, g=0.5018 (medium), CI [0.0046874, 0.0402342] sıfırı içermiyor.': 1 karşılaştırma
- Priority(E): 2 ağırlık ablasyonunda downstream verification_yield farkının %95 CI'si sıfırı dışlıyor — bu ağırlıkların nedensel etkisi gerçek.

## Sınırlar

- Tohum artırmak varyansı daha iyi tahmin ettirir; sistematik yanlılığı (veri üretimi, görev tasarımı) düzeltmez.
- Eşleşmiş testler aynı tohumun aynı veri havuzunu ürettiğini varsayar; protokoller bunu garanti eder, dışarıdan verilen seriler için doğrulanmamıştır.
- Çoklu karşılaştırma düzeltmesi (Bonferroni/FDR) uygulanmadı; çok sayıda kıyasta tek tek p-değerleri iyimserdir.
- Bootstrap CI küçük n'de asimptotik değildir; n=20'de aralıklar gerçek kapsamanın biraz altında kalabilir.
