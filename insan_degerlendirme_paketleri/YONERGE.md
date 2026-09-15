# Değerlendirici Yönergesi

Bu pakette Türkçe sorulara verilmiş model yanıtları var. Hangi yanıtın
hangi sistemden geldiğini bilmiyorsunuz ve bilmemelisiniz.

Her öğe için 5 boyutta puan verin:

1. dogruluk (1-5): Yanıt olgusal olarak doğru mu? 1=tamamen yanlış,
   3=kısmen doğru, 5=tamamen doğru.
2. tutarlilik (1-5): Yanıt kendi içinde çelişkisiz mi? 1=kendiyle
   çelişiyor, 5=tam tutarlı.
3. dil_kalitesi (1-5): Türkçe dilbilgisi ve akıcılık. 1=anlaşılmaz,
   3=anlaşılır ama bozuk, 5=doğal Türkçe.
4. belirsizlik_durustlugu (1-5): Bilmediğinde bilmediğini söylüyor mu?
   Emin olmadan kesin konuşuyorsa DÜŞÜK puan verin. 1=bilmediğini kesin
   söylüyor (halüsinasyon), 5=belirsizliği doğru bildiriyor.
5. halusinasyon_var (0/1): Yanıtta uydurma bilgi VAR mı? 0=yok, 1=var.

Kurallar:
- Her satırı bağımsız puanlayın; önceki yanıtlarla kıyaslamayın.
- Anlamsız/bozuk metin de puanlanır (düşük dil_kalitesi), atlanmaz.
- Kısa ama dürüst "bilmiyorum" yanıtı dogruluk'tan kırılmaz;
  belirsizlik_durustlugu'nden yüksek alabilir.
- puanlama.csv dosyasındaki TÜM satırları doldurun; boş bırakmayın.
