# Self-Learning Ölçeklendirme (P1)

- Protokol: `self_learning_scaling_v1` v1 (imza `745d4ee6ceca`)
- Profil / tohumlar: `deep` / `[1]`

## Tüm noktalar

| Cycle | Alan | Bilgi | Bilgi/cycle | EY | Yanlış | FAR | Süre (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 31 | 909 | 8.090 | 0.1264 | 0 | 0.0000 | 1.1 |
| 1000 | 31 | 938 | 0.838 | 0.1260 | 0 | 0.0000 | 10.9 |
| 3000 | 31 | 938 | 0.279 | 0.1260 | 0 | 0.0000 | 32.3 |
| 1000 | 63 | 3685 | 3.585 | 0.1246 | 0 | 0.0000 | 43.0 |
| 3000 | 127 | 14735 | 4.878 | 0.1248 | 0 | 0.0000 | 688.2 |

## Cycle ekseni (alan sabit) — doygunluk

| Alan | Cycle çarpanı | Bilgi çarpanı | Ölçekleme verimi | Doygun |
|---|---:|---:|---:|---|
| operands_max=31 | 30× | 1.032× | 0.0344 | EVET |

## Alan ekseni (cycle sabit) — gerçek ölçekleme

| Cycle | Alan çarpanı | Bilgi çarpanı |
|---|---:|---:|
| cycles=1000 | 2× | 3.93× |
| cycles=3000 | 4× | 15.71× |

## Sürüklenme (model collapse göstergesi)

- Toplam yanlış bilgi: `0`
- En uzun koşu: `3000` cycle → yanlış `0`, FAR `0.0`
- Doğrulayıcı ÖNCESİ en yüksek FAR: `1.0000` (doğrulayıcının gerçekten iş yaptığının kanıtı)

> Sürüklenme = uzun koşuda yanlış bilginin birikmesi. CLOSED_VERIFIED rejiminde doğrulayıcı bunu engellemelidir; sıfırdan büyük bir değer self-training çöküşünün başladığını gösterir.

## Kabul kapıları

| Kapı | Sonuç |
|---|---|
| all_points_completed | GEÇTİ |
| no_incorrect_knowledge_accumulated | GEÇTİ |
| no_false_acceptance_after_verifier | GEÇTİ |
| train_test_isolation_clean | GEÇTİ |
| long_run_stable | GEÇTİ |
| reached_1000_cycles | GEÇTİ |
| reached_3000_cycles | GEÇTİ |
| saturation_measured | GEÇTİ |
| domain_axis_measured | GEÇTİ |

## Bulgular

- Izgara: [(100, 31), (1000, 31), (3000, 31), (1000, 63), (3000, 127)], tohumlar [1] → 5 koşu.
- DOYGUNLUK (operands_max=31): cycle 100→3000 (30×) bilgiyi yalnız 909→938 (1.032×) artırdı. Ölçekleme verimi 0.0344. Sabit alanda aday havuzu tükeniyor; ek cycle hesap israfıdır. 'Daha çok döngü = daha çok bilgi' YANLIŞ.
-   └ Nihai bilginin %99'una 103. cycle'da ulaşıldı; kalan 2897 cycle marjinal.
- ALAN EKSENİ (cycles=1000): operands_max 2× büyüyünce bilgi 3.93× arttı. Ölçekleme cycle sayısından değil ALANIN genişliğinden geliyor.
- ALAN EKSENİ (cycles=3000): operands_max 4× büyüyünce bilgi 15.71× arttı. Ölçekleme cycle sayısından değil ALANIN genişliğinden geliyor.
- Sürüklenme yok: en uzun koşuda (3000 cycle) yanlış bilgi 0, doğrulayıcı sonrası FAR 0.0. CLOSED_VERIFIED rejimi uzun koşuda çökmüyor.
- Deneyim verimi (EY) tüm ölçeklerde 0.1246–0.1264 bandında kaldı; ölçek EY'yi değiştirmiyor, çünkü üretim dağılımı sabit.

## Sınırlar

- Alan aritmetiktir (a+b=c); doğrulayıcı kapalı formdur. Gerçek dünya alanlarında doğrulama bu kadar kesin olmaz.
- Doygunluk bu alanın SONLU olmasından gelir; sonsuz bir alanda cycle ölçeklemesi farklı davranabilir.
- 'Sürüklenme yok' sonucu CLOSED_VERIFIED rejimine özgüdür; doğrulayıcısız self-training ayrıca ölçülmelidir.
- Tek makine, tek süreç; paralel öğrenme davranışı kapsam dışı.
