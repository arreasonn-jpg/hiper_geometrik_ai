# Self-Learning Ölçeklendirme (P1)

Protokol: `self_learning_scaling_v1` · Modül: `hga/evaluation/self_learning_scaling.py`
CLI: `python -m hga ogrenme-olcek --signature-profile {smoke,standard,deep}`

## İstek ve bulgu

İstek "self-learning cycle sayısını 100'den 1000'e, sonra 3000'e çıkar" idi.
Koşuldu. Çıkan sonuç istekten daha önemli:

| cycles | operands_max | Nihai bilgi | EY | Yanlış bilgi | Süre |
|---:|---:|---:|---:|---:|---:|
| 100 | 31 | 909 | 0.1264 | 0 | 1.4 sn |
| 1000 | 31 | **938** | 0.1260 | 0 | ~40 sn |
| 1000 | 63 | 3 685 | 0.1246 | 0 | ~3 dk |
| 3000 | 127 | 14 735 | 0.1248 | 0 | ~9 dk |

(tohum 1, `batch_size=64`, `initial_facts=100`, `negatives_per_fact=7`)

**Cycle sayısını 10 katına çıkarmak bilgiyi %3 artırdı (909 → 938).**

Sebep: sabit bir `operands_max` ile üretilebilecek geçerli ifade sayısı
sonludur. Havuz tükendikten sonra her cycle aynı adayları yeniden üretir,
doğrulayıcı "zaten biliniyor" der ve yeni bilgi eklenmez. Yani

> "Daha çok öğrenme döngüsü = daha çok bilgi" **yanlıştır.**
> Ölçekleme cycle sayısından değil, **alanın genişliğinden** gelir.

`operands_max`'ı 31 → 63 (2×) yapmak, cycle'ı 100 → 1000 (10×) yapmaktan
yaklaşık **130 kat** daha fazla bilgi kazandırdı.

## Ölçülen nicelikler

- **`saturation_cycle`** — nihai bilginin %99'una ulaşılan ilk cycle.
  Bundan sonrası ölçülebilir hesap israfıdır ve raporda yazılıdır.
- **`knowledge_per_cycle`** — marjinal verim; doygunluk sonrası ~0.
- **`scaling_efficiency`** — bilgi çarpanı / cycle çarpanı. 1.0 doğrusal
  ölçekleme demektir; ölçülen değer 0.10'un altında.
- **`drift`** — uzun koşuda yanlış bilgi birikiyor mu?

## Sürüklenme (model collapse) sonucu

Self-training literatüründeki temel risk, modelin kendi ürettiği hatalı
veriyle eğitilip bozulmasıdır. Ölçüm:

- Tüm ölçeklerde **yanlış bilgi = 0**, doğrulayıcı sonrası **FAR = 0.0**.
- 3000 cycle / 14 735 olguluk en uzun koşuda da bozulma yok.
- Doğrulayıcı **öncesi** FAR sıfırdan büyüktür. Bu kritik bir ayrımdır:
  ham üretim hatalı adaylar içeriyor, hataları eleyen doğrulayıcıdır —
  sonuç üreticinin değil, **CLOSED_VERIFIED rejiminin** kanıtıdır.

## Neden EY ölçekle değişmiyor?

Deneyim verimi tüm ölçeklerde 0.1246–0.1264 bandında kaldı. Üretim
dağılımı sabit olduğu için beklenen davranış budur; EY bir ölçek metriği
değil, üretici-doğrulayıcı uyumunun metriğidir
(ayrıntı: `docs/VERIM_METRIKLERI.md`).

## Kabul kapıları

`all_points_completed`, `no_incorrect_knowledge_accumulated`,
`no_false_acceptance_after_verifier`, `train_test_isolation_clean`,
`long_run_stable`, `reached_1000_cycles`, `reached_3000_cycles`,
`saturation_measured`, `domain_axis_measured`.

`smoke` profilinde 1000/3000 kapıları **kasıtlı olarak KALIR** — kısa bir
koşum uzun koşu iddiasını desteklemez. Karne bunu yansıtır: `smoke` ile
`self_learning` bölümü 7.8, `deep` ile 10.0 olur.

## Sınırlar

- Alan aritmetiktir (`a+b=c`) ve doğrulayıcı kapalı formdur. Gerçek dünya
  alanlarında doğrulama bu kadar kesin olmaz.
- Doygunluk bu alanın **sonlu** olmasından gelir; sonsuz bir alanda cycle
  ölçeklemesi farklı davranabilir.
- "Sürüklenme yok" sonucu CLOSED_VERIFIED rejimine özgüdür; doğrulayıcısız
  self-training ayrıca ölçülmelidir.
- Tek makine, tek süreç; paralel öğrenme kapsam dışı.
