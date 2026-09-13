# Faz 27/28 — Köken (Provenance) · Faz 25 — Priority(E) Ağırlıkları

```bash
python -m hga koken      # köken denetimi + belge hash doğrulaması
python -m hga oncelik    # Priority(E) ağırlıkları + terim ablasyonu
```

Ham çıktı: `raporlar/provenance.json`, `raporlar/priority_ablation.json`
· Kod: `hga/evaluation/provenance.py`, `hga/experience/exploration.py`
· Testler: `tests/test_provenance.py` (11), `tests/test_priority_weights.py` (9)

---

## Faz 27/28 — "Nereden biliyorum?"

Bir bilgi tabanının bilimsel olarak savunulabilir olması için "biliyorum"
demesi yetmez. Milestone koşusunda ölçülen `C_V/C_E = 0.9965` yalnız aritmetik
oracle tam olduğu için yüksekti; gerçek metinde doğrulama oracle'ı yoktur ve
elimizdeki tek dayanak **kaynağın izlenebilirliğidir**.

### Şema değişikliği (geriye dönük uyumlu)

`RelationFact` beş opsiyonel köken alanı kazandı: `source_url`,
`document_hash`, `sentence`, `extractor`, `retrieved_at`. Hepsi `None`
varsayılanlıdır — mevcut çağrılar ve kayıtlı veriler etkilenmez.

`olgu_kaydet(..., provenance={...})` ile köken damgalanır; bilinmeyen alan
adı verilirse **açık hata** fırlatılır (sessizce yutulmaz).

### Denetim sözleşmesi

| Kaynak türü | Köken zorunlu mu? | Gerekçe |
|---|:--:|---|
| `REAL_DATA`, `EXTERNAL_VERIFIED`, `HUMAN_CONFIRMED` | **evet** | dış belgeye dayanır |
| `VERIFIED_RULE`, `DERIVED`, `MODEL_GENERATED`, `FREE_GENERATION` | hayır | kökeni kural/model, dış belge değil |

`audit_provenance(store)` dış kaynaklı her olguda `source_url` **ve**
`document_hash` arar; eksikse "yetim olgu" sayar. `assert_clean()` eksiklikte
hata verir.

### Belge değişikliği yakalanıyor

`verify_document_hashes(store, {url: içerik})` kaydedilen hash'i belgenin
gerçek SHA-256'sıyla karşılaştırır. Demo koşusunda belge sonradan
değiştirildiğinde **2/2 olgu** "içerik değişmiş" olarak işaretlendi. Belge hiç
sağlanmazsa bu da ayrı raporlanır (`unknown_documents`) — doğrulanamayan bir
iddia, doğrulanmış sayılmaz.

### Asıl darboğaz: ayıklama verimi

`ingest_with_provenance` cümleleri **tek tek** aktarır; böylece her olgu kendi
ham cümlesine kesin olarak bağlanır (toplu aktarımda bu eşleme kaybolur, çünkü
ayıklayıcı `entity_id` değil token döndürür). Bu yol `extraction_yield`
metriğini de ortaya çıkardı:

> Sözlük tabanlı `CumleAyiklayici`, sözlükte olmayan kelimeler içeren cümleleri
> **atlıyor**. Demo korpusunda 4 cümleden 2'si üçlüye dönüştü
> (`extraction_yield = 0.5`).

Bu, Faz 27/28'in gerçek sınırıdır: köken altyapısı hazır, ama gerçek Türkçe
veriye geçmek için ayıklayıcının kapsamı darboğaz. Yeni ayıklayıcı yazmak
feature freeze kapsamı dışında bırakıldı — **ölçüldü ve raporlandı**.

---

## Faz 25 — Priority(E) ağırlıkları açık ve ayarlanabilir

    Priority(E) = w_gain·InfoGain + w_novelty·Novelty
                  + w_uncertainty·Uncertainty − w_conflict_penalty·Conflict

### Ne değişti

1. **Tek kaynak**: `VARSAYILAN_PRIORITY_AGIRLIKLARI` ve
   `PRIORITY_AGIRLIK_ACIKLAMALARI` sözlükleri; her ağırlığın ne ölçtüğü yazılı.
2. **Doğrulama**: negatif ağırlık reddedilir (bir sinyali devre dışı bırakmak
   için `0.0` kullanılır; ceza terimi zaten ayrı ve çıkarılır).
3. **Normalizasyon**: `normalize=True` pozitif ağırlıkları 1'e ölçekler, farklı
   ağırlık setleri karşılaştırılabilir olur.
4. **Şeffaflık**: `priority_dokumu()` her terimin değerini, ağırlığını ve
   katkısını ayrı ayrı döndürür.
5. **Dürüstlük düzeltmesi**: eski kod `except Exception` ile *her* hatayı
   yutup `gain=novelty=0.5` atıyordu. Artık yalnız `KeyError` yakalanıyor ve
   bu bir hata değil **epistemik durum** olarak ele alınıyor: kayıt yoksa aday
   tamamen yenidir (`novelty = uncertainty = 1.0`). Gerçek hatalar artık
   gizlenmiyor.

### Ağırlık ablasyonu — asıl bulgu

`agirlik_ablasyonu()` her terimi tek tek kapatıp seçilen ilk-K kümesinin ne
kadar değiştiğini ölçer:

| Kapatılan terim | Seçim örtüşmesi | Değişen seçim | Etkili mi |
|---|---:|---:|:--:|
| `w_gain` | 0.700 | 3 | evet |
| `w_conflict_penalty` | 0.900 | 1 | evet |
| `w_novelty` | 1.000 | 0 | **HAYIR** |
| `w_uncertainty` | 1.000 | 0 | **HAYIR** |

**Okunuşu:** `w_novelty` ve `w_uncertainty` ağırlıklarını sıfırlamak, seçilen
ilk 10 adayı **hiç değiştirmedi**. Yani bu iki terim bu veri üzerinde
işlevsizdir. Ağırlığın "ayarlanabilir" olması, onun **etkili** olduğu anlamına
gelmez — ve bu ayrımı ancak ablasyon gösterebilir.

Bu, düzeltilmiş bir hata değil, **ölçülmüş bir zaaftır**: mevcut sentetik veri
üzerinde Priority(E)'nin dört teriminden ikisi seçimi yönlendirmiyor. Gerçek
veride (ya da daha çeşitli aday havuzunda) davranış değişebilir; önemli olan
artık bunun ölçülebilir olmasıdır.
