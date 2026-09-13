# Sparse Memory Collision ve Interference Benchmarkı

Bu benchmark `hga.memory.DeneyimSlotlari` saf Python prototipini ölçer. Neural
`HashlenmisKureselTablo` için doğrudan kalite sonucu olarak yorumlanmamalıdır.

## Ölçülen değerler

Birbirinden farklı `N` context sabit `S` slota yazılır. Context listesi RAM'de
tutulmaz; seed ve sıra numarasından ikinci geçişte yeniden üretilir.

```text
collision_event_rate = collision_events / (N × table_count)
retrieval_accuracy    = retained_contexts / N
interference_rate     = (N - retained_contexts) / N
false_positive_rate   = wrong_id_matches / probes
```

Ek olarak yazma/okuma throughput'u, dolu slot sayısı, tablo başına collision,
hiçbir tam context tarafından geri çağrılamayan `orphaned_slots`, yaklaşık
CPython depolama boyutu ve collision örneklerinin kırpılıp kırpılmadığı
raporlanır.

## Interference politikası

Mevcut politika `first-writer-wins` şeklindedir:

1. Boş slota ilk deneyim yazılır.
2. Aynı slota gelen farklı deneyim collision olarak sayılır.
3. İlk deneyim overwrite edilmez.
4. Yeni deneyim o adreste geri çağrılamaz.

Bu nedenle collision, mevcut kaydı bozmak yerine **yeni kayıt kaybı** üretir.
Benchmark bunu `interference_loss` olarak görünür kılar.

## Çift tablo hakkında doğrulanan sınır

`tablo_sayisi=2` modunda deneyim iki tabloya yazılır ve exact-ID okuma iki
tablonun da eşleşmesini (`all`) ister. Dolayısıyla ikinci tablo ilk tablodaki
kaybı telafi edemez. İki tablodan sağ kalan küme, ilk tablodan sağ kalanların
alt kümesidir:

```text
retrieval_accuracy(table_count=2) <= retrieval_accuracy(table_count=1)
```

Ayrıca fiziksel slot bütçesi iki katına çıkar. Bu sonuç çift tablonun gereksiz
olduğunu tek başına kanıtlamaz; mevcut Bloom-benzeri exact-membership
semantiğinin collision recovery sağlamadığını gösterir. Alternatif iki-seçimli,
bucket veya collision-chain tasarımları ancak bu baseline'a karşı ayrı
benchmark ile değerlendirilmelidir.

## Kullanım

Hızlı koşu:

```bash
python -m hga memory-benchmark \
  --scales 1000,10000,100000 \
  --slots 65536 --tables 1,2 --seeds 1,2,3,4,5
```

Talep edilen tam ölçek:

```bash
python -m hga memory-benchmark \
  --scales 1000,10000,100000,1000000,10000000 \
  --slots 1048576 --tables 1,2 --seeds 1,2,3,4,5 \
  --experiment-root experiments \
  --out raporlar/memory_stress_summary.json
```

10M × 5 seed × 2 tablo CPU üzerinde uzun sürebilir. Runner her seed için
commit, config/dataset hash'i, ortam, stdout ve sonucu ayrı `EXP-NNNN`
dizininde kaydeder. Varsayılan CLI bu nedenle yalnız 1K/10K/100K çalıştırır.

## İlk kontrollü baseline

Koşul: `slot_count=65,536`, seed `42`, sentetik context generator v1. Bu tablo
performans süresini değil deterministik collision/retrieval sonuçlarını verir.
Tek seed olduğu için güven aralığı yoktur; genel dağılım iddiası değildir.

| N | Tablo | Collision event rate | Retrieval accuracy | Interference rate |
|---:|---:|---:|---:|---:|
| 1,000 | 1 | 0.00600 | 0.99400 | 0.00600 |
| 1,000 | 2 | 0.00800 | 0.99000 | 0.01000 |
| 10,000 | 1 | 0.07060 | 0.92940 | 0.07060 |
| 10,000 | 2 | 0.10400 | 0.86260 | 0.13740 |
| 100,000 | 1 | 0.48666 | 0.51334 | 0.48666 |
| 100,000 | 2 | 0.58727 | 0.31212 | 0.68788 |

Bu baseline mevcut ALL-table politikasının yüksek yükte retrieval'ı ciddi
biçimde düşürdüğünü ve ikinci tablonun recovery sağlamadığını doğrular. Sonuç
bir tasarım değişikliğinin gerekçesidir; henüz alternatif tasarımın üstünlüğü
kanıtlanmış değildir.

## Yorumlama kuralları

- Aynı fiziksel bütçe karşılaştırılacaksa iki tablo için tablo başına slot
  sayısı yarıya indirilmelidir.
- Throughput sonucu cihaz ve Python sürümüyle birlikte okunmalıdır.
- Yaklaşık depolama metriği RSS değildir.
- Sentetik context testi gerçek dil dağılımının kanıtı değildir.
- Düşük false-positive oranı tek başına başarı değildir; retrieval ve
  interference birlikte raporlanmalıdır.
