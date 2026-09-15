# tr_corpus_v1 — Milyon-Kelime Gerçek Türkçe LM Korpusu

**1.110.509 kelime, 4.365 belge, 104.230 cümle.** Tamamı gerçek, insan
üretimi Türkçe metin; sentetik veya makine çevirisi içerik YOKTUR.
`corpus.jsonl.gz` içindeki her satır `{doc_id, source, text}` alanlı bir
JSON belgesidir; `text` satır başına bir cümledir.

## Kaynaklar

| Kaynak | Belge | Kelime | Lisans | Pin |
|---|---:|---:|---|---|
| UD_Turkish-Kenet | 468 | 178.658 | CC BY-SA 4.0 | r2.14 |
| UD_Turkish-Penn | 410 | 183.555 | CC BY-SA 4.0 | r2.14 |
| UD_Turkish-BOUN | 245 | 101.036 | CC BY-SA 4.0 | r2.14 |
| UD_Turkish-Tourism | 496 | 91.152 | CC BY-SA 4.0 | r2.14 |
| UD_Turkish-Atis | 136 | 45.907 | CC BY-SA 4.0 | r2.14 |
| UD_Turkish-FrameNet | 70 | 19.223 | CC BY-SA 4.0 | r2.14 |
| UD_Turkish-PUD | 25 | 14.415 | CC BY-SA 3.0 | r2.14 |
| UD_Turkish-GB | 73 | 14.177 | CC BY-SA 4.0 | r2.14 |
| bible-corpus (Turkish.xml) | 66 | 450.598 | CC0 1.0 | commit SHA |
| TWT v1 (repoda vendored) | ~2.4K | ~66K | Apache-2.0 | 40838e5 |

Kesin sayılar ve tüm SHA-256 hash'leri `PROVENANCE.json`'dadır (dedup
sonrası kelime toplamları tablodakinden hafifçe düşebilir).

## Bilinçli kararlar

- **UD_Turkish-IMST DIŞARIDA**: CC BY-NC-SA (NonCommercial) lisansı bu
  deponun yeniden dağıtım profiliyle uyumsuz.
- **Belge birimi**: Bible → kitap (66 belge); TWT → doğal `sent_id` belge
  öneki; UD → ardışık 40 cümlelik deterministik sözde-belge (UD dosyaları
  doğal belge sınırı yayınlamaz). LM split'i belge-ayrık olduğu için belge
  birimi split sızıntısını belirler; tek dev belge kabul edilemezdi.
- **Dedup**: korpus geneli normalize-exact (NFKC + casefold + boşluk)
  cümle tekilleştirme; 7.514 tekrar cümle düşürüldü.
- **Çapraz-görev notu**: TWT aynı zamanda `twt_real_results_v1` arc
  doğrulama benchmarkının verisidir. Görevler farklıdır (arc etiketi vs
  next-token LM); bu ilişki gizlenmez, burada ve PROVENANCE'ta belgelidir.
- **Tür dengesi dürüstlüğü**: İncil çevirisi korpusun ~%40'ıdır. Modern
  Türkçe çeviridir ama tematik olarak dardır; `turkish_lm_v1` raporu bu
  sınırlılığı `limitations` bölümünde açıkça taşır.

## Yeniden üretim ve doğrulama

```bash
python hga/data/build_tr_corpus.py   # ağ gerektirir; pinli revizyonlardan kurar
```

Yükleyici (`hga.evaluation.turkish_lm._load_tr_corpus_raw_documents`) hem
gzip dosya SHA-256'sını hem gzip'ten bağımsız canonical içerik SHA-256'sını
doğrular; uyuşmazlıkta korpus YÜKLENMEZ.
