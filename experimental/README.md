# experimental/ — Entegre Edilmemiş Kodlar

Bu klasördeki dosyalar **ana eğitim/çıkarım akışına bağlı değildir** ve hiçbir
modül tarafından import edilmezler. Eski sürümde bu dosyalar `mimari/` altında
duruyor ama kullanılmıyorlardı; şimdi durumları açıkça belgelenmiştir.

## bpe_tokenizer.py

Alternatif bir **alt-kelime (subword/BPE)** tokenizer. Ana akış bunun yerine
kelime düzeyli `mimari/tokenizer.py` (`GeometrikTokenizer`) kullanır.

- BPE'yi ana akışa taşımak isterseniz: `tokenizer_hazirla()` (ortak.py) içinde
  tokenizer sınıfını değiştirmeniz ve modelleri yeniden eğitmeniz gerekir;
  sözlük kimlikleri tamamen değişir.
- Kendi kendine çalıştırılabilir örnek:
  `python -c "from experimental.bpe_tokenizer import BPETokenizer; t = BPETokenizer().fit_on_text(open('turkce_metin.txt', encoding='utf-8').read()); print(t.sozluk_boyutu)"`

## encoder.py

İlk tasarımdan kalma, **512 boyutlu ham veri vektörü** bekleyen bir encoder.
Mevcut token-tabanlı embedding akışıyla uyumsuzdur (girdi boyutu farklı);
"sanal matris" deneyi için tutulmuştur.
