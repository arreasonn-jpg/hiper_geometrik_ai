# -*- coding: utf-8 -*-
import json
import os

ZENGIN = [
    {"soru": "merhaba", "cevap": "merhaba ben hiper geometrik yapay zekayım size nasıl yardımcı olabilirim"},
    {"soru": "merhaba nasılsın", "cevap": "merhaba iyiyim teşekkür ederim size nasıl yardımcı olabilirim"},
    {"soru": "selam", "cevap": "selam hoş geldiniz size nasıl yardımcı olabilirim"},
    {"soru": "günaydın", "cevap": "günaydın harika bir gün geçirmenizi dilerim"},
    {"soru": "sen kimsin", "cevap": "ben hiper geometrik fraktal yapay zekayım türkçe sohbet için tasarlandım"},
    {"soru": "adın ne", "cevap": "adım hiper geometrik fraktal ai"},
    {"soru": "neler yapabilirsin", "cevap": "soruları cevaplarım bilgi veririm ve türkçe metin üretebilirim"},
    {"soru": "teşekkür ederim", "cevap": "rica ederim her zaman yardımcı olmaktan mutluluk duyarım"},
    {"soru": "görüşürüz", "cevap": "görüşmek üzere kendinize iyi bakın"},
    {"soru": "duygu nedir", "cevap": "duygu insanın dış dünyadaki olaylara karşı hissettiği psikolojik tepkilerdir"},
    {"soru": "yapay zeka nedir", "cevap": "yapay zeka insan zekasını taklit eden öğrenen ve problem çözen bilgisayar sistemleridir"},
    {"soru": "tesseract nedir", "cevap": "tesseract üç boyutlu küpün dört boyutlu karşılığı olan hiper küptür"},
    {"soru": "fraktal nedir", "cevap": "fraktal her ölçekte kendine benzeyen karmaşık geometrik şekillerdir"},
    {"soru": "bilim nedir", "cevap": "bilim evrendeki olguların deney ve gözlemle sistematik incelenmesidir"},
    {"soru": "felsefe nedir", "cevap": "felsefe varlık bilgi ve değer konularını derinlemesine inceleyen düşünce disiplinidir"},
    {"soru": "teknoloji nedir", "cevap": "teknoloji bilimin hayatı kolaylaştırmak için uygulamaya dönüşmesidir"},
    {"soru": "türkiyenin başkenti neresidir", "cevap": "türkiye cumhuriyetinin başkenti ankaradır"},
    {"soru": "türkiyenin başkenti neresi", "cevap": "türkiyenin başkenti ankaradır"},
    {"soru": "başkent neresi", "cevap": "türkiyenin başkenti ankaradır"},
    {"soru": "ankara nedir", "cevap": "ankara türkiye cumhuriyetinin başkentidir"},
    {"soru": "en kalabalık şehir hangisi", "cevap": "türkiyenin en kalabalık şehri istanbuldur"},
    {"soru": "cumhuriyet ne zaman ilan edildi", "cevap": "türkiye cumhuriyeti yirmi dokuz ekim bin dokuz yüz yirmi üçte ilan edildi"},
    {"soru": "atatürk kimdir", "cevap": "mustafa kemal atatürk türkiye cumhuriyetinin kurucusu ve ilk cumhurbaşkanıdır"},
    {"soru": "istanbul ne zaman fethedildi", "cevap": "istanbul bin dört yüz elli üç yılında fatih sultan mehmet tarafından fethedildi"},
    {"soru": "en büyük gezegen hangisi", "cevap": "güneş sistemindeki en büyük gezegen jüpiterdir"},
    {"soru": "ışık hızı ne kadardır", "cevap": "ışık hızı saniyede yaklaşık üç yüz bin kilometredir"},
    {"soru": "su kaç derecede kaynar", "cevap": "su deniz seviyesinde yüz derecede kaynar"},
    {"soru": "beş ile yedinin çarpımı", "cevap": "beş ile yedinin çarpımı otuz beştir"},
    {"soru": "nasıl çalışırsın", "cevap": "hiper küresel fraktal mimari ve çok kafalı dikkat ile türkçe metin işlerim"},
    {"soru": "ne yapıyorsun", "cevap": "sizinle sohbet ediyorum ve sorularınızı cevaplıyorum"}
]

class TalimatToplayici:
    def __init__(self, dosya_yolu="talimat_verisi.json"):
        self.dosya_yolu = dosya_yolu
    def hazirla_veya_yukle(self):
        # Düzeltme: artık gerçekten YÜKLÜYOR — eskiden dosya var olsa bile
        # her çalıştırmada üzerine yazılıyordu (sözlük/eğitim sessizce kayardı).
        if os.path.exists(self.dosya_yolu):
            try:
                with open(self.dosya_yolu, "r", encoding="utf-8") as f:
                    veri = json.load(f)
                if isinstance(veri, list) and veri:
                    print(f"✅ Talimat seti diskten yüklendi: {len(veri)} örnek")
                    return veri
            except Exception:
                pass
        with open(self.dosya_yolu, "w", encoding="utf-8") as f:
            json.dump(ZENGIN, f, ensure_ascii=False, indent=2)
        print(f"✅ Talimat seti oluşturuldu: {len(ZENGIN)} örnek")
        return ZENGIN
