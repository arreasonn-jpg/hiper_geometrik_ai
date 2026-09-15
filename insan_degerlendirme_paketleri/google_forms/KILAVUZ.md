# Google Forms ile kör insan değerlendirmesi

Bu dizin, mevcut kör paketleri Google Forms'a taşır ve indirilen yanıtları
HGA'nın mevcut `Rxx_puanlama.csv` şemasına geri çevirir. Entegrasyon **insan
puanı üretmez**; gerçek, yetkin değerlendiricilerden veri gelmeden
`human_evaluation` bölümü `n/a` kalmalıdır.

| Dosya | İşlev |
|---|---|
| `paket_forma.gs` | Google Apps Script: Drive'daki `Rxx.json` paketlerinden bölünmüş Forms üretir. |
| `forms_yaniti_donustur.py` | Forms/Sheets CSV dışa aktarımlarını doğrular ve `Rxx_puanlama.csv` biçimine birleştirir. |

## Güvenlik ve körleme sınırı

- Değerlendiricilere yalnız kendi Forms bağlantılarını ve yönergeyi verin.
- **Asla** `_GIZLI_degerlendiriciye_verme/kor_anahtari.json` dosyasını,
  `_GIZLI_degerlendiriciye_verme/` dizinini veya Drive klasörünü paylaşmayın.
  Bu anahtar `item_id → kol` eşlemesini içerir; sızarsa körleme geçersiz olur.
- Script, `paketler/R01.json` gibi kamuya giden kör dosyaları okur; kol adı
  oluşturmaz veya göstermemelidir. Form başlıklarındaki `HGA|…` dizeleri yalnız
  makine-okunur `item_id`/puan boyutu anahtarlarıdır.
- Rater kodu (`R01`) **kimlik doğrulaması değildir**. Her formu yalnız atanmış
  Google hesabıyla paylaşın; “bağlantıya sahip herkes” erişimini açmayın.
- Google Forms/Sheets üçüncü taraf barındırmadır. Kurumun etik kurul, KVKK,
  onam, saklama süresi ve erişim politikalarına uyun. Yanıtlarda kişisel veri
  istemeyin; Forms'un e-posta toplama ayarını ancak onam/protokol gerektiriyorsa
  etkinleştirin.

## 1. Drive hazırlığı

1. Google Drive'da iki **ayrı** klasör oluşturun:
   - `HGA_forms_girdi`: yalnız repo içindeki
     `insan_degerlendirme_paketleri/paketler/R01.json` … `R10.json` dosyalarını
     yükleyin. CSV şablonlarını ve gizli kör açma dizinini yüklemeyin.
   - `HGA_forms_cikti`: oluşturulan Forms, yanıt e-tabloları ve kayıt dosyası
     burada dursun. Bu klasör yalnız araştırma ekibine açık kalmalıdır.
2. Her klasörün URL'sindeki `/folders/` sonrasındaki kimliği kopyalayın.
3. `paket_forma.gs` dosyasını [script.google.com](https://script.google.com)'da
   yeni bir Apps Script projesine yapıştırın. Dosyanın üstündeki `CONFIG`
   içinde `PACKAGE_FOLDER_ID` ve `OUTPUT_FOLDER_ID` yer tutucularını bu
   kimliklerle değiştirin. `ITEMS_PER_FORM=40` değerini değiştirmeyin.
4. Projeyi kaydedin. İlk çalıştırmada Google Drive, Forms ve Sheets izinleri
   istenir; bunlar yalnız sizin proje hesabınıza verilmelidir.

## 2. Formları oluşturma ve paylaşma

Apps Script düzenleyicisinden önce `createNextRaterForms` çalıştırın. Bu,
örneğin `R01` için 200 öğeyi beş bölüme ayırır. Her bölümde 40 öğe vardır:

- her öğenin soru/yanıt metni ve beş zorunlu puanı,
- bir `R01` rater-kodu alanı,
- ayrı bir bağlı Google Sheets yanıt dosyası.

Bölme kasıtlıdır: 200 öğe × beş puan tek Form'da platformun öğe sınırlarını ve
kullanılabilirliği zorlar. Beş parçanın tümünde sabit `HGA|item_id|boyut`
başlıkları bulunduğu için Python aracı sonuçları tekrar tek CSV'de birleştirir.

`Logger` çıktısındaki `HGA_google_forms_registry.json` kaydını saklayın. Bu
kayıtta düzenleme/yayın URL'leri, bağlı e-tablolar, kaynak paket SHA-256'sı ve
hangi öğelerin hangi bölümde olduğu bulunur; **kol eşlemesi bulunmaz**.

- Önce bir deneme hesabıyla yalnız bir rater paketi üzerinde uçtan uca test
  yapın. Formlarda görünen metinde `hga`, `dense`, `transformer` veya
  `symbolic` gibi kol etiketleri olmadığını kontrol edin.
- Her bölüm formunda ayarlardan **yanıtları sınırla** / atanmış hesabı seçin.
  İsterseniz “bir yanıtla sınırla”yı açın. Kimlik bilgisini araştırma planı
  gerektirmedikçe toplamayın.
- Değerlendiriciye kendi beş form bağlantısını, kör yönergeyi ve rater kodunu
  gönderin. Yanıtı değiştirme veya yeniden gönderme sürecini önceden belirtin.
- Aynı scripti yeniden çalıştırmak mevcut `rater_id` kaydını atlar. Bilerek
  yeni tur başlatılacaksa ayrı bir output klasörü ve ayrı Apps Script projesi
  kullanın; önceki turu üzerine yazmayın.

`createAllRemainingForms` tüm eksik paketleri oluşturur. Apps Script süre
kotası nedeniyle yarıda kalırsa yeniden çalıştırın; her tamamlanan rater
kayıttan sonra kaydedilir ve tekrar üretilmez.

## 3. Yanıtları dışa aktarma

Her bölüm formunun bağlı yanıt e-tablosunda:

1. Yanıtlar sayfasından **Dosya → İndir → Virgülle ayrılmış değerler (.csv)**
   seçin.
2. Dosya adını değiştirebilirsiniz; dönüştürücü ad yerine sütun şemasını ve
   rater kodunu denetler.
3. Bütün bölüm CSV'lerini boş bir yerel klasöre koyun. Analizden önce Forms'ta
   beklenen tüm beş bölümün yanıtlandığını kontrol edin.

CSV başlıklarındaki `HGA|...` alanlarını elle değiştirmeyin. Hatalı rater kodu,
bilinmeyen `item_id`, 1–5 dışı ordinal puan, 0/1 dışı halüsinasyon puanı,
eksik hücre veya çifte yanıt varsayılan olarak **hata** üretir. Sessiz veri
kaybı yoktur.

## 4. Dönüştürme ve bütünlük denetimi

Repo kökünden çalıştırın:

```bash
python insan_degerlendirme_paketleri/google_forms/forms_yaniti_donustur.py \
  --responses ~/Downloads/hga_forms_csv \
  --packages insan_degerlendirme_paketleri \
  --output /tmp/hga_forms_puanlari \
  --summary /tmp/hga_forms_import.json
```

Başarılı çıktıda `/tmp/hga_forms_puanlari/R01_puanlama.csv` … dosyaları,
mevcut şablonla aynı sütun sırasındadır:

```csv
item_id,dogruluk,tutarlilik,dil_kalitesi,belirsizlik_durustlugu,halusinasyon_var
4927523800f73815,1,2,1,4,0
```

`--summary` dosyası kaynak CSV'leri, görülen satırları, rater başına
beklenen/mevcut/eksik hücre sayısını ve paket dosyası SHA-256'sını tutar.
Bunu ham yanıtlarla birlikte saklayın. Ara teslim için zorunlu olarak
`--allow-partial` eklenebilir; araç uyarı yazar ve bu çıktı güvenilirlik/sonuç
olarak yorumlanmamalıdır.

Bir değerlendirici bir bölümü bilerek tekrar gönderdiyse varsayılan hata yerine
zaman damgasına göre son satırı seçmek için açıkça şunu kullanın:

```bash
... forms_yaniti_donustur.py ... --duplicate-policy latest
```

Bu karar ve nedenini araştırma günlüğüne kaydedin. `latest` yalnız Forms'un
`Timestamp` alanı güvenilir olduğunda kullanılmalıdır.

## 5. Mevcut HGA analizine aktarma

İçe aktarılan **tam** dosyaları ayrı, erişimi kısıtlı bir kopyadaki
`paketler/` dizinine koyun. Orijinal kör paket şablonlarını ve ham Forms
CSV'lerini saklayın; kör açma anahtarını hâlâ kapalı tutun.

```python
from pathlib import Path
from hga.evaluation.human_eval_fill import collect_ratings_from_csv
from hga.evaluation.human_evaluation import build_human_evaluation_protocol

ratings, collection = collect_ratings_from_csv(Path("guvenli_degerlendirme_kopyasi"))
report = build_human_evaluation_protocol(collected_ratings=ratings)
print(collection)
print(report.results)  # Boyut başına Krippendorff α
```

Mevcut toplayıcı en az on dolu değerlendirici CSV'si olmadan insan-puanı
kapısını açmamalıdır. Ayrıca α ≥ 0.800 güvenilirlik eşiği sağlanmadan sonuç
raporlanamaz. `α`, kodlayıcı uyumunu ölçer; dış dünya doğruluğunu tek başına
kanıtlamaz.

## Sınırlar

- Google Forms aktarımı anket lojistiğini çözer; temsilî örneklem, rater
  eğitimi, onam, dikkat kontrolü, bağımsız doğruluk denetimi veya istatistiksel
  sonuç üretmez.
- Paketler aynı rater kodunu tekrar görür. Bu tanımlayıcı pseudonymous bir
  operasyon kodudur; kimlik/erişim denetiminin yerine geçmez.
- Forms arayüzü ve CSV yerelleştirmesi değişebilir. Dönüştürücü yalnız
  değişmez `HGA|...` başlıklarını varsayar; deneme turu her platform değişiminde
  tekrarlanmalıdır.
