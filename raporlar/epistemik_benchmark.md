# Epistemik Benchmark (P0-007)

Veri kümesi hash: `34e88fbfcacae3ded071312f898966e0beaa665b6fdc3c6a85815538a9021b21`

- Yanlış güven oranı (↓): **0.000**
- Bilinmeyen doğruluğu (↑): **1.000**
- Sessiz kabul oranı (↓): **0.000**
- Genel doğruluk: **1.000**

## Sınıf bazında

| Sınıf | n | Doğru | İsabet |
|---|---:|---:|---:|
| KNOWN | 8 | 8 | 1.000 |
| FALSE | 8 | 8 | 1.000 |
| UNKNOWN | 5 | 5 | 1.000 |
| UNCERTAIN | 4 | 4 | 1.000 |
| CONFLICT | 3 | 3 | 1.000 |

## Epistemik çözünürlük

UNKNOWN (kayıt hiç yok) ve UNCERTAIN (özellik yazılmamış) epistemik olarak farklıdır. Durum KODU ikisini de DeneyimDurumu.UNCERTAIN'e indirger (distinguishable_by_state=False) fakat ExperienceCandidate.belirsizlik_sebebi alanı kaynağı ayırır: KAYIT_YOK ('bilmiyorum') ve OZELLIK_YOK ('emin değilim'). Ayrım durum makinesi geçişleri bozulmadan ölçülebilir.

## Sınırlar

- Kontrollü ontoloji üzerinde epistemik karar testidir; gerçek dil anlama değildir.
- UNKNOWN ve UNCERTAIN tek duruma eşlendiği için ayrım ölçülür, varsayılmaz.
- Sonuç Evaluator karar ağacının davranışıdır; neural üretim kalitesi değildir.
- false_confidence_rate yalnız bu fixture içindir; genel halüsinasyon oranı değildir.
