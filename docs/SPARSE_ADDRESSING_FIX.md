# Seyrek Bellek Adresleme Düzeltmesi — "Bloom tarzı" İki Tablonun Sıfır Bağımsızlığı

Modül: `hga/memory/sparse_memory.py` (`DeneyimSlotlari`)
Etki alanı: multi-hop çıkarım (`reasoning_depth`, `depth_diagnosis`), bellek
benchmarkları (`hga/memory/benchmark.py`), Engine deneyim deposu.

## Kusur

`DeneyimSlotlari` iki tabloyu "Bloom tarzı" bağımsız hash diye sunuyordu.
Gerçekte adresleme şuydu:

```
adres(iz, tablo_t) = (iz · t) mod N        # t = 1, 2
```

Üç ayrı kusur:

1. **Sıfır bağımsızlık.** `iz₁ ≡ iz₂ (mod N)` ise `2·iz₁ ≡ 2·iz₂ (mod N)`.
   Tablo-1'de çakışan HER anahtar çifti tablo-2'de de çakışır. Ölçüm:
   200.000 anahtarda tablo-1'de çakışan 46.817 grubun **46.817'si** (hepsi)
   tablo-2'de de çakıştı. İkinci tablo hiçbir şey katmıyordu.
2. **Yarı kapasite.** N çift iken (tüm profiller 2^k kullanır) `(iz·2) mod N`
   yalnız ÇİFT slotlara düşer: tablo-2'nin tek numaralı slotları hiçbir
   zaman kullanılmadı (50.000 anahtarda 41.598 adres, 0 tek sayılı).
3. **AND okuma kaybı büyütür.** `icerir()` deneyimi TÜM tablolarda arıyordu.
   Slot içeriği `experience_id` ile karşılaştırıldığı için yanlış-pozitif
   zaten imkânsızdır; AND semantiği hiçbir yanlış-pozitifi engellemiyor,
   yalnızca tek tablodaki kaybı toplam kayba çeviriyordu. Kayıp olasılığı
   bağımsız tablolarda OR ile p² olur; AND ile ≈ 2p.

**Ölçülen etki:** 2^18 slotta (%0.2 doluluk!) 513 anahtardan 3'ü kayboluyordu;
512-hop çıkarım zinciri kopuyordu. `REASONING_DEPTH_ROOT_CAUSE.md`'nin
ölçtüğü "bellek kapasitesi sınırı" büyük ölçüde bu adresleme kusurunun
eseriydi — teşhisin *yöntemi* doğruydu (çöküş gerçekten bellekteydi,
çıkarımda değildi), ama *sınırın kendisi* gerçek kapasite değil, kusurdu.

## Düzeltme

1. Tablo başına bağımsız adres: parmak izi tablo tuzuyla XOR'lanıp splitmix
   çığ karışımından geçirilir (`_TABLO_TUZLARI`, `_adres`). Ölçüm: tablo-1'de
   çakışan 46.760 grubun tablo-2'de de çakışanı **11** (≈ beklenen rastgele
   oran); tablo-2 artık tüm slotları kullanıyor.
2. `icerir()` OR semantiğine geçti: deneyim herhangi bir tabloda hayatta ise
   bulunur. Exact-ID karşılaştırması yanlış-pozitifi imkânsız kılmayı sürdürür.

`yaz()` değişmedi: first-writer-wins, çakışma sayacı, LRU aynen durur.

## Ölçülen sonuç (düzeltme öncesi → sonrası)

| Koşul | Önce | Sonra |
|---|---|---|
| 513 anahtar / 2^18 slot kaybı | 3 | 0 |
| 2.049 anahtar / 2^18 slot kaybı | 15 | 0 |
| 32 hop @ 16.384 dolgu, 2^18 slot | KIRIK (C_RD=8) | temiz |
| C_R (deep profil, 2^18 slot, ızgara-içi kırılma) | 32 (tavan) | **2.048 (ölçüm)** |
| Desteklenen derinlik @ 16.384 dolgu | 8 → slot başına doğrusal | 512 @2^18 → 2.048 @2^19 → 4.096+ @2^20 |
| log-log eğim (derinlik ∝ slot^eğim) | 1.000 | **2.0** |

Eğim değişimi tesadüf değil, kusurla tutarlı fizik: eski adreslemede iki
tablo tek tablo gibi davranıyordu → kenar kaybı p ∝ doluluk → derinlik ∝ N
(eğim 1). Bağımsız tablolar + OR ile kayıp p² → derinlik ∝ N² (eğim 2).
Eski "eğim 1.000" bulgusunun kendisi kusurun parmak iziydi.

## Nelere dokunuldu

- `hga/memory/sparse_memory.py` — `_adres` (bağımsız tuzlar), `icerir` (OR).
- `hga/memory/benchmark.py` — policy metni + not: ANY okuma semantiği.
- `hga/evaluation/reasoning_depth.py` — profiller yeni kırılma bölgelerine
  göre yeniden boyutlandırıldı; C_R artık her profilde ızgara-içi ölçümdür.
- `hga/evaluation/depth_diagnosis.py` — teşhis ızgaraları dolgu-baskın
  tasarıma taşındı (D >> max hop; kontrol kolu her slotta temiz kalır).
- `tests/test_memory_benchmark.py`, `tests/scientific/test_depth_diagnosis.py`,
  `tests/scientific/test_reasoning_depth.py` — eski kusurlu semantiği
  "sözleşme" olarak kodlayan beklentiler gerçek davranışa güncellendi.

## Dürüstlük notları

- Bu düzeltme çıkarım YETENEĞİNİ değiştirmez; zincir takibi hep doğruydu
  (dolgu=0 kontrol kolu hiç bozulmamıştı). Değişen şey bellek katmanının
  kayıp oranıdır.
- `retains_half_depth_under_max_distractors` kapısı deep profilde hâlâ
  FAIL'dir (16.384 dolguda 2.048 → 256, tutma 0.125): 2^18 slot bütçesinde
  dolgu baskısı gerçek bir sınırdır ve gizlenmez. Kök neden teşhisi bunun
  bellek olduğunu göstermeyi sürdürür.
- Eski sayıları (C_R=32→256, eğim 1.000) alıntılayan belgeler güncellendi;
  düzeltme commit'i öncesi üretilmiş raporlar o commit'in kodu için
  geçerlidir ve yeniden yorumlanmaz.
