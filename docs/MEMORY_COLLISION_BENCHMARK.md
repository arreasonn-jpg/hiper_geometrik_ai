# Sparse Memory Collision ve Interference Benchmarkı

Legacy mod `hga.memory.DeneyimSlotlari` FIRST_WINS tablosunu ölçer. Aynı
`memory-benchmark` komutunun `--active-dynamic` modu ayrıca Engine'de gerçekten
kullanılan bounded `DynamicKVMemory` implementasyonunu aynı unique streaming
akışta kırar. Neural `HashlenmisKureselTablo` için doğrudan kalite sonucu
olarak yorumlanmamalıdır.

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

Aktif v2 stress protokolünde iki bounded depo aynı fiziksel entry capacity ile
tek akışta ilerler:

```text
fixed_exact_history_recall   = occupied_slots / written_contexts
dynamic_exact_history_recall = active_records / written_contexts
dynamic_active_sample_recall = retained audit hits / retained audit probes
```

İlk iki oran tahmin değildir. Table-1 FIRST_WINS'te her dolu slot yalnız bir
retained kimlik, unique-key bounded Dynamic KV'de her aktif kayıt yalnız bir
retained kimlik taşıdığı için fiziksel cardinality tam history muhasebesidir.
Deterministik audit hem aktif son pencerenin `1.0` recall'ını hem evicted eski
pencerenin reddini kontrol eder. Girdi corpus'u/listesi materyalize edilmez.
Journal event ve lazy-LRU heap uzunlukları capacity'nin sabit katıyla bounded
olmak zorundadır. `/proc/self/status` RSS ve yaklaşık CPython storage ayrı
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

Talep edilen tam aktif ölçek:

```bash
python -m hga memory-benchmark --active-dynamic \
  --scales 1000,10000,100000,1000000,10000000 \
  --slots 65536 --audit-samples 256 --seeds 1,2,3,4,5 \
  --experiment-root experiments \
  --out raporlar/memory_stress_1k_10m.json \
  --markdown raporlar/memory_stress_1k_10m.md
```

V2 her seed'de 10M'e bir kez yürür; küçük checkpoint'ler aynı akıştan alınır.
Yine de 10M × 5 seed CPU üzerinde uzun sürebilir. Runner her seed için commit,
config/dataset hash'i, fiziksel capacity, ortam, stdout ve sonucu ayrı
`EXP-NNNN` dizininde kaydeder. Varsayılan CLI bu nedenle yalnız
1K/10K/100K ve legacy modu çalıştırır.

## Tam aktif v2 sonucu (1K→10M, 5 seed)

Temiz commit `94fe3f6757d6b8b08e240f11862e4795daae6aea`, capacity `65.536`, audit
sample `256`, seedler `1,2,3,4,5`. Dataset hash
`c46da1ee227948e65743b42980bd3c7e57b9ac8fed5f5bda11a6b93065a1a5d2`,
config hash `2ef4f1d96e7d0e4e5312d3716da38159306fb4de7c43bc9d9cf1acbfc887bd6f`.
Tüm manifestler clean/COMPLETED ve tüm per-seed kapılar geçti. Ham/aggregate
sonuç: `raporlar/memory_stress_1k_10m.json`; okunur tablo:
`raporlar/memory_stress_1k_10m.md`.

| Context | Fixed history recall | Dynamic history recall | Dynamic active recall | Eviction | ctx-pair/s |
|---:|---:|---:|---:|---:|---:|
| 1K | 0.991600±0.002154 | 1.000000±0 | 1.000000±0 | 0 | 92.207±5.707 |
| 10K | 0.926120±0.000773 | 1.000000±0 | 1.000000±0 | 0 | 86.162±9.326 |
| 100K | 0.512728±0.001186 | 0.655360±0 | 1.000000±0 | 34.464 | 60.163±4.281 |
| 1M | 0.065536±0 | 0.065536±0 | 1.000000±0 | 934.464 | 27.277±1.749 |
| 10M | 0.006554±0 | 0.006554±0 | 1.000000±0 | 9.934.464 | 27.409±564 |

10M'de fixed ve bounded Dynamic history recall eşitliği aynı hata mekanizması
demek değildir: fixed ilk 65.536 hash slot sahibini collision ile korur;
Dynamic KV son 65.536 tam anahtarı LRU ile korur. Dynamic aktif pencerenin
sample recall'ı beş seed'de de `1.0`, evicted sample rejection `1.0` ve tam
muhasebe `active + evictions = writes` olarak geçti. Buna rağmen tüm tarih
recall'ı `65.536 / 10.000.000 = 0.0065536`'dır. Bounded belleğin eski bilgiyi
koruduğu iddia edilemez.

Process-içi RSS 10M checkpoint'inde `577.30±1.01 MiB` oldu; bu değer PyTorch'u
manifest metadata için yükleyen uzun ömürlü runner ve CPython allocator
retention'ını da içerir, yalnız KV payload'ı değildir. CPython storage tahmini
ayrı raporlanır. Seed süreleri 357–373 saniyedir.

## İlk kontrollü legacy baseline

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
- `dynamic_active_sample_recall=1.0`, tüm geçmişin korunduğu anlamına gelmez;
  history recall capacity/context oranıyla ayrıca düşer.
- `segment_context_pairs_per_second`, her context için bir fixed ve bir Dynamic
  KV yazısını birlikte içerir; tek-depo throughput değildir.
- Throughput sonucu cihaz ve Python sürümüyle birlikte okunmalıdır.
- Yaklaşık depolama metriği RSS değildir; ikisi ayrı raporlanır.
- Sentetik context testi gerçek dil dağılımının kanıtı değildir.
- Düşük false-positive oranı tek başına başarı değildir; retrieval ve
  interference birlikte raporlanmalıdır.
