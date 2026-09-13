# Faz 15–17 — Memory Interference: Kasıtlı Çakışma ve Sabit/Dinamik Ödünleşim

Üretim komutu:

```bash
python -m hga memory-interference --slots 4096 --forced-collisions 50 \
    --out raporlar/memory_interference.json
```

Ham çıktı: `raporlar/memory_interference.json` · Kod: `hga/memory/interference.py`
· Testler: `tests/test_memory_interference.py` (10 test)

## Neden bu deney?

Mevcut `hga/memory/benchmark.py` collision'ı **istatistiksel** ölçüyordu: N
context yaz, kaçı çakıştı? Düşük yük faktöründe çakışma neredeyse hiç olmadığı
için bu ölçüm "sparse memory sorunsuz" yanılsaması üretmeye müsaitti.

100 döngülük milestone koşusu (`docs/MILESTONE_TABLOSU.md`) bu yanılsamayı
kırdı: **memory recall 1.000 → 0.952, collision 0 → 19**. Sistemin 100 döngü
boyunca sınırı verifier değil, **sabit slotlu sparse memory** oldu. Bu modül o
çatlağın mekanizmasını izole eder: çakışmayı beklemek yerine **kasıtlı olarak
kurar**.

```
A → slot 123
B → slot 123          (adversarial arama ile seçilmiş)
Soru: A bilgisi B tarafından bozuluyor mu?
```

## Sonuç 1 — Saklama politikası kıyası (50 kasıtlı çakışma, 4096 slot)

| Politika | Kurban yaşar | Saldırgan yaşar | İkisi birden | Bozulma | Fiziksel girdi | Bayt |
|---|---:|---:|---:|---:|---:|---:|
| `FIRST_WINS` | 1.000 | 0.000 | 0.000 | 0.000 | 50 | 6,642 |
| `LAST_WINS` | 0.000 | 1.000 | 0.000 | 1.000 | 50 | 6,742 |
| `DYNAMIC_KV` | 1.000 | 1.000 | 1.000 | 0.000 | 100 | 13,544 |

**Okunuşu:**

- `FIRST_WINS` (mevcut davranış) kurbanı korur ama yeni bilgiyi **sessizce
  düşürür** — saldırgan yaşama oranı 0.000. Sistem "hata vermiyor" ama
  öğrenmiyor.
- `LAST_WINS` yeni bilgiyi alır, **eski doğrulanmış bilgiyi bozar** (bozulma
  1.000). Doğrulanmış bilgi tabanı için kabul edilemez.
- İki sabit-tablo politikasında da `both_survival = 0.000`. Bu bir ayar/eşik
  meselesi **değil**, sabit adres uzayının yapısal sonucudur: iki anahtar aynı
  slota düşüyorsa, tek slot iki kimliği taşıyamaz.
- `DYNAMIC_KV` her ikisini korur, bedeli girdi başına büyüyen depolamadır.

## Sonuç 2 — Sabit tablo vs dinamik KV ölçekleme (4096 slot, dim=32)

| Context | Yük faktörü | Sabit recall | Collision | Dinamik recall | Önceden tahsisli tablo (bayt) | Dinamik KV (bayt) | Dinamik/Tahsisli |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.0244 | 0.9800 | 2 | 1.0000 | 524,288 | 13,494 | 0.026 |
| 1,000 | 0.2441 | 0.8900 | 110 | 1.0000 | 524,288 | 125,790 | 0.240 |
| 10,000 | 2.4414 | 0.3739 | 6,261 | 1.0000 | 524,288 | 1,193,898 | 2.277 |
| 100,000 | 24.4141 | 0.0410 | 95,904 | 1.0000 | 524,288 | 14,330,276 | 27.333 |

**Okunuşu:**

- Sabit tablonun recall'ı yük faktörüyle birlikte **çöker**: yük 24× olduğunda
  recall 0.041. Yani 4096 slotlu bir tabloya 100 bin context yazmak, bilginin
  %96'sını kaybetmek demektir.
- Dinamik KV recall'ı tanım gereği 1.000'dir; ödediği bedel RAM'dir.
- **Dönüm noktası ~4.000–10.000 context arası**: bu ölçeğin altında dinamik KV
  hem daha doğru hem daha ucuz (oran < 1). Üstünde ödünleşim gerçek hâle gelir
  — 100 bin contextte dinamik KV 27× daha fazla bellek ister.

### Ölçüm dürüstlüğü uyarısı

`fixed_storage_bytes` sütunu rapordan **çıkarıldı**, çünkü saf Python "sabit"
tablo aslında bir `dict`'tir ve önceden tahsis etmez — bu, gerçek
`nn.Embedding` davranışını temsil etmez. Karşılaştırmanın geçerli sütunu
`preallocated_table_bytes`'tır: `slot × tablo × boyut × 4 bayt`, context
sayısından **bağımsız** sabit maliyet.

Bu modül saf Python prototip ölçümüdür. PyTorch `HashlenmisKureselTablo`
toplamsal vektör okuması yapar; oradaki bozulma "kimlik kaybı" değil "vektör
karışması" biçiminde görünür ve ayrıca ölçülmelidir.

## Karar için çıkarım

Mevcut mimari 4096 slotla **birkaç bin doğrulanmış bilgiye kadar** güvenlidir.
Bunun ötesinde üç seçenek vardır ve hiçbiri bedava değildir:

1. Slot sayısını büyüt — sabit RAM maliyeti, ölçek yine sonlu.
2. Dinamik KV'ye geç — recall 1.0, RAM veriyle büyür.
3. Melez: sıcak bilgi dinamik KV'de, soğuk bilgi sabit tabloda (LRU zaten
   `lru_temizle` ile mevcut). Bu yön **henüz ölçülmedi**.

Ayrıca `collision_events` metriği tek başına yetersizdir: çakışma sayısı
düşükken bile *hangi* bilginin kaybolduğu önemlidir. Bu yüzden
`victim_survival_rate` / `corruption_rate` ayrı raporlanır.
