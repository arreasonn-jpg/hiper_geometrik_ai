# -*- coding: utf-8 -*-
"""Türkçe korpus toplayıcı: Hugging Face veri setleri + Wikipedia.

DÜRÜST NOT — üçüncü taraf bağımlılık: Varsayılan veri kaynağı, kişisel bir
Hugging Face veri seti reposudur (``VARSAYILAN_REPO``). Bu repo silinir,
gizlenir veya yapısı değişirse veri toplama akışı kırılabilir. Farklı bir
kaynak için ``--repo`` argümanını veya ``--konu`` (Wikipedia) seçeneğini
kullanın:

    python -m egitim.veri_toplayici --repo baska/bir_dataset --kelime 40000
    python -m egitim.veri_toplayici --konu "Türkiye" --konu "Ankara"
"""
import argparse
import csv
import io
import json
import logging
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq
import requests

log = logging.getLogger("hiper.veri")

# Varsayılan üçüncü taraf veri kaynağı (bkz. README "Veri Kaynağı" bölümü).
VARSAYILAN_REPO = "ShigeoKageyama/NLP_SUITE"


class OtomatikVeriToplayici:
    def __init__(self, proje_kok, repo_id=VARSAYILAN_REPO):
        self.proje_kok = proje_kok
        self.wiki_api = "https://tr.wikipedia.org/w/api.php"
        self.hf_api = "https://huggingface.co/api/datasets"
        self.headers = {
            "User-Agent": "HiperGeometrikAI/10.0 (Smart Stream-Batch Parquet Engine)"
        }
        self.desteklenen_uzantilar = (
            ".txt", ".text", ".md", ".markdown",
            ".json", ".jsonl", ".csv", ".tsv", ".parquet"
        )
        self.durum_dosyasi = os.path.join(proje_kok, "yama_durumu.json")
        self.durum = self._durum_yukle()
        self.durum["repo"] = self.durum.get("repo", repo_id)

    # ==========================================
    # YAMA DURUM YÖNETİMİ
    # ==========================================
    def _durum_yukle(self):
        if os.path.exists(self.durum_dosyasi):
            try:
                with open(self.durum_dosyasi, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "repo": VARSAYILAN_REPO,
            "islenen_dosyalar": [],
            "son_patch": 0,
            "toplam_cekilen_kelime": 0
        }

    def _durum_kaydet(self):
        with open(self.durum_dosyasi, "w", encoding="utf-8") as f:
            json.dump(self.durum, f, ensure_ascii=False, indent=2)

    def yama_raporu(self):
        islenen = self.durum.get("islenen_dosyalar", [])
        log.info(f"\n[SİSTEM] 📦 YAMA DURUM RAPORU")
        log.info(f"  Repo: {self.durum.get('repo')}")
        log.info(f"  Son işlenen patch no: {self.durum.get('son_patch', 0)}")
        log.info(f"  İşlenen dosya sayısı: {len(islenen)}")
        log.info(f"  Toplam çekilen kelime: {self.durum.get('toplam_cekilen_kelime', 0)}")
        if islenen:
            log.info(f"  Son 3 dosya:")
            for d in islenen[-3:]:
                log.info(f"    - {d}")

    # ==========================================
    # KALİTE SÜZGECİ
    # ==========================================
    def _temizle_metin(self, metin):
        metin = re.sub(r'\[\d+\]', '', str(metin))
        metin = re.sub(r'https?://\S+', '', metin)
        metin = re.sub(r'www\.\S+', '', metin)
        metin = re.sub(r'\s+', ' ', metin)
        return metin.strip()

    def _kaliteli_mi(self, metin):
        if not metin or len(metin) < 40:
            return False
        kelimeler = metin.split()
        if len(kelimeler) < 8:
            return False
        rakam_orani = sum(1 for k in kelimeler if any(c.isdigit() for c in k)) / max(1, len(kelimeler))
        if rakam_orani > 0.35:
            return False
        tr_karakter = sum(1 for c in metin.lower() if c in "abcçdefgğhıijklmnoöprsştuüvyz ")
        if tr_karakter / max(1, len(metin)) < 0.55:
            return False
        return True

    def _json_icerik_ayikla(self, veri):
        metinler = []
        if isinstance(veri, dict):
            if "soru" in veri and "cevap" in veri:
                metinler.append(f"{veri['soru']} {veri['cevap']}")
            elif "instruction" in veri and "output" in veri:
                girdi = veri.get("input", "")
                metinler.append(f"{veri['instruction']} {girdi} {veri['output']}")
            elif "prompt" in veri and "response" in veri:
                metinler.append(f"{veri['prompt']} {veri['response']}")
            elif "question" in veri and "answer" in veri:
                metinler.append(f"{veri['question']} {veri['answer']}")
            elif "text" in veri and isinstance(veri["text"], str):
                metinler.append(veri["text"])
            elif "conversations" in veri and isinstance(veri["conversations"], list):
                for tur in veri["conversations"]:
                    val = tur.get("value", tur.get("content", ""))
                    if val:
                        metinler.append(val)
            else:
                for _, v in veri.items():
                    metinler.extend(self._json_icerik_ayikla(v))
        elif isinstance(veri, list):
            for eleman in veri:
                metinler.extend(self._json_icerik_ayikla(eleman))
        elif isinstance(veri, str):
            if len(veri.strip()) > 3:
                metinler.append(veri.strip())
        return metinler

    def format_donusturucu(self, dosya_adi, ham_icerik):
        uzanti = os.path.splitext(dosya_adi)[1].lower()
        try:
            if uzanti == ".json":
                veri = json.loads(ham_icerik)
                ham = ". ".join(self._json_icerik_ayikla(veri))
            elif uzanti in [".csv", ".tsv"]:
                ayirici = "\t" if uzanti == ".tsv" else ","
                okuyucu = csv.reader(io.StringIO(ham_icerik), delimiter=ayirici)
                satirlar = [" ".join([h.strip() for h in r if len(h.strip()) > 1]) for r in okuyucu]
                ham = ". ".join(satirlar)
            else:
                ham = re.sub(r'#+\s*', '', ham_icerik)
                ham = re.sub(r'[*_`~]', '', ham)
            return self._temizle_metin(ham)
        except Exception:
            return ""

    # ==========================================
    # HF DOSYA LİSTELEME (SAYFALAMA + RECURSIVE)
    # ==========================================
    def _hf_dosyalari_listele(self, repo_id):
        """
        Hugging Face dataset tree API:
        - recursive=1
        - cursor ile tüm sayfalar
        - klasörleri de elle tarar (iç içe parquet için)
        """
        dosyalar = []
        gorulen_path = set()

        def _tree_sayfalari(path_prefix=""):
            """Belirli bir path altındaki tüm öğeleri (sayfalı) getir."""
            cursor = None
            alt_klasorler = []
            sayfa = 0
            while True:
                sayfa += 1
                if path_prefix:
                    url = f"{self.hf_api}/{repo_id}/tree/main/{path_prefix}?recursive=1"
                else:
                    url = f"{self.hf_api}/{repo_id}/tree/main?recursive=1"
                if cursor:
                    url += f"&cursor={cursor}"

                try:
                    yanit = requests.get(url, headers=self.headers, timeout=45)
                except Exception as e:
                    log.warning(f"  [⚠️ HF TREE] {e}")
                    break

                if yanit.status_code != 200:
                    # recursive bazen path'li URL'de fail olabilir → recursive'siz dene
                    if path_prefix and "recursive" in url:
                        url2 = f"{self.hf_api}/{repo_id}/tree/main/{path_prefix}"
                        if cursor:
                            url2 += f"?cursor={cursor}"
                        try:
                            yanit = requests.get(url2, headers=self.headers, timeout=45)
                        except Exception:
                            break
                    if yanit.status_code != 200:
                        if sayfa == 1 and not path_prefix:
                            log.warning(f"  [⚠️ HF TREE HTTP {yanit.status_code}] fallback siblings...")
                        break

                try:
                    data = yanit.json()
                except Exception:
                    break

                if not isinstance(data, list):
                    break

                if not data:
                    break

                for oge in data:
                    p = oge.get("path") or oge.get("rpath") or ""
                    if not p or p in gorulen_path:
                        continue
                    gorulen_path.add(p)
                    tip = (oge.get("type") or "").lower()
                    # directory
                    if tip in ("directory", "folder") or oge.get("type") == "directory":
                        alt_klasorler.append(p)
                        continue
                    dosyalar.append(p)

                # Sonraki sayfa cursor (header veya JSON)
                cursor = None
                link = yanit.headers.get("Link") or yanit.headers.get("link") or ""
                # <...&cursor=XXX>; rel="next"
                m = re.search(r'cursor=([^&>]+)[^>]*>;\s*rel="next"', link)
                if m:
                    cursor = m.group(1)
                else:
                    # bazı yanıtlarda son öğede cursor olmaz; HF xet/cursor header
                    cursor = yanit.headers.get("X-Cursor") or yanit.headers.get("x-cursor")
                    if not cursor:
                        break

                if sayfa > 200:  # güvenlik freni
                    log.warning("  [⚠️ HF TREE] 200 sayfa limitine ulaşıldı")
                    break

            return alt_klasorler

        # 1) Kök recursive tarama
        log.info("  [📂 HF] Dosya ağacı taranıyor (recursive + sayfalama)...")
        alt = _tree_sayfalari("")

        # 2) Fallback: siblings API
        if not dosyalar and not alt:
            try:
                yanit = requests.get(f"{self.hf_api}/{repo_id}", headers=self.headers, timeout=20)
                if yanit.status_code == 200:
                    data = yanit.json()
                    if isinstance(data, dict) and "siblings" in data:
                        for f in data["siblings"]:
                            p = f.get("rpath") or f.get("path") or ""
                            if p:
                                dosyalar.append(p)
            except Exception as e:
                log.warning(f"  [⚠️ siblings] {e}")

        # 3) Hâlâ az dosya / sadece üst dizin: patch-8 tarzı iç içe klasörleri elle gez
        #    Özellikle AI-Training-Patch-N/data/<alt>/*.parquet
        patch_kokleri = set()
        for p in list(dosyalar) + list(alt) + list(gorulen_path):
            m = re.search(r"(AI-Training-Patch-\d+)", p, re.IGNORECASE)
            if m:
                patch_kokleri.add(m.group(1))

        # Bilinen patch klasörlerini de ekle (liste boşsa bile 1..20 dene — opsiyonel hafif)
        if not patch_kokleri:
            # tree hiç patch göstermediyse kullanıcı filtresi için 1-15 arası yokla
            for i in range(1, 16):
                patch_kokleri.add(f"AI-Training-Patch-{i}")

        for pk in sorted(patch_kokleri):
            # data ve alt klasörler
            for prefix in (pk, f"{pk}/data"):
                alt2 = _tree_sayfalari(prefix)
                # data altındaki her alt klasör (aym_bb, yargitay, ...)
                for klasor in alt2:
                    if klasor.count("/") >= 2:  # Patch-x/data/yargitay
                        _tree_sayfalari(klasor)

        # Sadece desteklenen uzantıları ayıklama ana fonksiyonda yapılıyor
        dosyalar = sorted(set(dosyalar))
        log.info(f"  [📂 HF] Toplam listelenen dosya: {len(dosyalar)}")
        return dosyalar

    def _patch_no_bul(self, dosya_yolu):
        m = re.search(r"AI-Training-Patch-(\d+)", dosya_yolu, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return 0

    # ==========================================
    # 🎯 RANGE READER: 25 GB+ DOSYALAR İÇİN
    # ==========================================
    class _HTTPRangeReader:
        """HTTP Range Request ile parquet dosyasını sanal olarak açar."""
        def __init__(self, url, total_size, footer_start, footer_data, headers):
            self.url = url
            self.total_size = total_size
            self.footer_start = footer_start
            self.footer_data = footer_data
            self.headers = headers
            self.pos = 0
            self._closed = False
            self.indirilen_toplam_bayt = len(footer_data)

        def seek(self, offset, whence=0):
            if whence == 0:
                self.pos = offset
            elif whence == 1:
                self.pos += offset
            elif whence == 2:
                self.pos = self.total_size + offset
            self.pos = max(0, min(self.pos, self.total_size))
            return self.pos

        def tell(self):
            return self.pos

        def read(self, size=-1):
            if size == -1 or size is None:
                size = self.total_size - self.pos
            if size <= 0:
                return b""

            end = min(self.pos + size, self.total_size)

            if self.pos >= self.footer_start:
                rel_start = self.pos - self.footer_start
                rel_end = end - self.footer_start
                data = self.footer_data[rel_start:rel_end]
                self.pos = end
                return data

            hdrs = dict(self.headers)
            hdrs["Range"] = f"bytes={self.pos}-{end - 1}"
            try:
                resp = requests.get(self.url, headers=hdrs, timeout=30)
                if resp.status_code in (200, 206):
                    self.pos = end
                    self.indirilen_toplam_bayt += len(resp.content)
                    return resp.content
            except Exception as e:
                log.warning(f"    [⚠️ RANGE READ] {e}")
            return b""

        def close(self):
            self._closed = True

        @property
        def closed(self):
            return self._closed

        def readable(self):
            return True

        def seekable(self):
            return True

        def writable(self):
            return False

    # ==========================================
    # SÜTUN BULUCU YARDIMCI (TÜRKÇE ÖNCELİKLİ)
    # ==========================================
    def _en_iyi_metin_sutunlarini_sec(self, pf):
        """Parquet dosyasındaki sadece TÜRKÇE metin sütunlarını akıllıca keşfeder."""
        try:
            arrow_schema = pf.schema_arrow
            schema_names = list(arrow_schema.names)
        except Exception:
            schema_names = list(pf.schema.names)

        # 1. Yabancı dilleri (en, fr, de, ru) baştan filtrele
        yabanci_kaliplar = ["_en", "english", "_de", "_fr", "_ru", "_es", "_ar"]
        turkce_adaylar = [c for c in schema_names if not any(y in c.lower() for y in yabanci_kaliplar)]

        # 2. Öncelikli Türkçe metin anahtarları
        oncelikli_anahtarlar = [
            "abstract_tr", "ozet", "tez_adi", "icerik", "title_tr", 
            "baslik", "metin", "aciklama", "text", "content", "konu"
        ]

        secilenler = []
        for anahtar in oncelikli_anahtarlar:
            for c in turkce_adaylar:
                if anahtar == c.lower() or f"_{anahtar}" in c.lower() or f"{anahtar}_" in c.lower():
                    if c not in secilenler:
                        secilenler.append(c)

        # 3. Bulunamadıysa genel Türkçe adaylardan ID olmayanları seç
        if not secilenler:
            secilenler = [c for c in turkce_adaylar if not any(stop in c.lower() for stop in ["id", "_no", "kod", "tarih", "yil", "date", "index"])]

        # 4. En fazla en önemli 2 sütunu al (RAM ve Ağ tasarrufu için)
        if len(secilenler) > 2:
            secilenler = secilenler[:2]

        return secilenler if secilenler else schema_names[:1]

    # ==========================================
    # 🔥 STREAMING PARQUET MOTORU (ANINDA CEVAP)
    # ==========================================
    def _parquet_akilli_oku(self, raw_url, hedef_kelime, mevcut_kelime):
        toplanan = []
        ek_kelime = 0

        try:
            head = requests.head(raw_url, headers=self.headers, timeout=15, allow_redirects=True)
            total_size = int(head.headers.get("Content-Length", 0))

            if total_size == 0:
                total_size = 500 * 1024 * 1024

            size_mb = total_size / (1024 * 1024)
            log.info(f"    [📏 BOYUT] {size_mb:.1f} MB")

            # 1. Footer'dan Şemayı Çek
            footer_boyut = 4 * 1024 * 1024  # 4 MB footer yeterlidir
            range_start = max(0, total_size - footer_boyut)
            headers_range = dict(self.headers)
            headers_range["Range"] = f"bytes={range_start}-{total_size - 1}"

            log.info(f"    [📥 METADATA] Footer okunuyor...")
            r = requests.get(raw_url, headers=headers_range, timeout=25)

            if r.status_code not in (200, 206):
                log.warning(f"    [⚠️] Range request desteklenmiyor (HTTP {r.status_code})")
                return "", 0

            footer_data = r.content
            reader = self._HTTPRangeReader(
                raw_url, total_size, range_start, footer_data, self.headers
            )

            try:
                pa_file = pa.PythonFile(reader, mode="r")
                pf = pq.ParquetFile(pa_file)

                log.info(f"    [📦 METADATA] Toplam: {pf.metadata.num_rows:,} satır")

                hedef_sutunlar = self._en_iyi_metin_sutunlarini_sec(pf)
                log.info(f"    [📋 SÜTUNLAR] Seçilen Türkçe Metin Sütunları: {hedef_sutunlar}")
                log.info(f"    [⚡ CANLI AKIŞ] Satırlar taranıyor (Hedef: +{hedef_kelime - mevcut_kelime} kelime)...")

                toplam_islenen_satir = 0

                # Satır gruplarını stream/batch olarak tara
                for batch in pf.iter_batches(batch_size=500, columns=hedef_sutunlar):
                    df_batch = batch.to_pandas()
                    toplam_islenen_satir += len(df_batch)

                    for col in hedef_sutunlar:
                        if col not in df_batch.columns:
                            continue
                        for val in df_batch[col].dropna().astype(str):
                            t = self._temizle_metin(val)
                            if self._kaliteli_mi(t):
                                toplanan.append(t)
                                ek_kelime += len(t.split())
                                if mevcut_kelime + ek_kelime >= hedef_kelime:
                                    break
                        if mevcut_kelime + ek_kelime >= hedef_kelime:
                            break

                    log.info(f"      ↳ {toplam_islenen_satir:,} satır işlendi | +{ek_kelime:,} kaliteli kelime süzüldü")

                    if mevcut_kelime + ek_kelime >= hedef_kelime:
                        log.info(f"    [🎯 HEDEF TAMAMLANDI] Yeterli kelimeye ulaşıldı, akış durduruldu.")
                        break

                indirilen_mb = reader.indirilen_toplam_bayt / (1024 * 1024)
                tasarruf_orani = max(0.0, 100 * (1 - indirilen_mb / size_mb))
                log.info(f"    [💾 TASARRUF] {size_mb:.1f} MB dosyadan sadece {indirilen_mb:.2f} MB veri çekildi! (%{tasarruf_orani:.2f} tasarruf)")

            except Exception as e:
                log.warning(f"    [⚠️ AKIŞ OKUMA HATASI] {e}")
                return "", 0
            finally:
                try:
                    reader.close()
                except Exception:
                    pass

            return " ".join(toplanan), ek_kelime

        except Exception as e:
            log.warning(f"    [⚠️ PARQUET GENEL HATASI] {e}")
            return "", 0

    # ==========================================
    # ANA DOSYA OKUMA (parquet + jsonl + diğer)
    # ==========================================
    def _dosya_oku_ve_ayikla(self, repo_id, dosya_yolu, hedef_kelime, mevcut_kelime):
        uzanti = os.path.splitext(dosya_yolu)[1].lower()
        raw_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{dosya_yolu}"

        if uzanti == ".parquet":
            return self._parquet_akilli_oku(raw_url, hedef_kelime, mevcut_kelime)

        elif uzanti == ".jsonl":
            toplanan = []
            ek_kelime = 0
            try:
                with requests.get(raw_url, headers=self.headers, stream=True, timeout=40) as r:
                    if r.status_code == 200:
                        for satir in r.iter_lines(decode_unicode=True):
                            if not satir or not satir.strip():
                                continue
                            try:
                                veri = json.loads(satir)
                                for m in self._json_icerik_ayikla(veri):
                                    t = self._temizle_metin(m)
                                    if self._kaliteli_mi(t):
                                        toplanan.append(t)
                                        ek_kelime += len(t.split())
                            except Exception:
                                continue
                            if mevcut_kelime + ek_kelime >= hedef_kelime:
                                break
            except Exception as e:
                log.warning(f"    [⚠️ JSONL HATASI] {dosya_yolu}: {e}")
                return "", 0
            return (" ".join(toplanan), ek_kelime)

        else:
            toplanan = []
            ek_kelime = 0
            try:
                res = requests.get(raw_url, headers=self.headers, timeout=25)
                if res.status_code == 200:
                    t = self.format_donusturucu(dosya_yolu, res.text)
                    parcalar = re.split(r'(?<=[\.!?])\s+', t)
                    for p in parcalar:
                        p = self._temizle_metin(p)
                        if self._kaliteli_mi(p):
                            toplanan.append(p)
                            ek_kelime += len(p.split())
                            if mevcut_kelime + ek_kelime >= hedef_kelime:
                                break
            except Exception as e:
                log.warning(f"    [⚠️ DOKÜMAN HATASI] {dosya_yolu}: {e}")
                return "", 0
            return (" ".join(toplanan), ek_kelime)

    # ==========================================
    # YAMA ÇEKME VE İŞLEME
    # ==========================================
    def huggingface_patch_cek(self, repo_id=VARSAYILAN_REPO, patch_secim="next",
                               mevcut_metin="", hedef_kelime_limiti=40000):
        repo_id = repo_id.strip().rstrip("/")
        repo_id = re.sub(r"^https?://huggingface\.co/datasets/", "", repo_id)
        self.durum["repo"] = repo_id

        log.info(f"\n  [🤗 MULTI-PATCH ENGINE] Repo: '{repo_id}' | Seçim: {patch_secim}")
        log.info(f"  [🛡️ AKILLI MOTOR] Range Request + Streaming Batch Aktif")

        try:
            tum_dosyalar = self._hf_dosyalari_listele(repo_id)
        except Exception as e:
            log.error(f"  [❌ LİSTE HATASI] {e}")
            return mevcut_metin

        uygun = [p for p in tum_dosyalar if os.path.splitext(p)[1].lower() in self.desteklenen_uzantilar]

        if not uygun:
            log.warning("  [⚠️] Desteklenen dosya bulunamadı.")
            return mevcut_metin

        if patch_secim not in ("next", "all"):
            try:
                istenen = int(str(patch_secim).strip())
                uygun = [p for p in uygun if self._patch_no_bul(p) == istenen]
                log.info(f"  [🎯 FİLTRE] Sadece AI-Training-Patch-{istenen} → {len(uygun)} dosya")
                if not uygun:
                    ipucu = [p for p in tum_dosyalar if self._patch_no_bul(p) == istenen
                             or f"Training-Patch-{istenen}" in p]
                    # uzantısız path'ler de (klasör)
                    log.info(f"  [🔍] Ham tree'de Patch-{istenen} ile eşleşen path: {len(ipucu)}")
                    for x in ipucu[:10]:
                        log.info(f"      • {x}")
            except ValueError:
                log.error("  [❌] Geçersiz patch no.")
                return mevcut_metin

        islenen_set = set(self.durum.get("islenen_dosyalar", []))
        uygun.sort(key=lambda x: (self._patch_no_bul(x), x))

        if patch_secim == "next":
            uygun = [p for p in uygun if p not in islenen_set]
            if not uygun:
                log.info("  [✅] Tüm bilinen yama dosyaları daha önce işlenmiş!")
                self.yama_raporu()
                return mevcut_metin

        indirilen = []
        toplam_kelime = 0
        yeni_islenen = []

        for dosya_yolu in uygun:
            if dosya_yolu in islenen_set and patch_secim == "next":
                continue

            patch_no = self._patch_no_bul(dosya_yolu)
            log.info(f"  [⬇️ Patch-{patch_no}] {dosya_yolu}")

            metin_parca, ek = self._dosya_oku_ve_ayikla(
                repo_id, dosya_yolu, hedef_kelime_limiti, toplam_kelime
            )

            if ek > 0 and metin_parca:
                indirilen.append(metin_parca)
                toplam_kelime += ek
                yeni_islenen.append(dosya_yolu)
                self.durum["son_patch"] = max(self.durum.get("son_patch", 0), patch_no)
                log.info(f"    [✅ KALİTELİ] +{ek} kelime (süzgeçten geçti)")
            else:
                yeni_islenen.append(dosya_yolu)
                log.info(f"    [⏭️ ATLANDI] Kaliteli metin yok veya pas geçildi")

            if toplam_kelime >= hedef_kelime_limiti:
                log.info(f"\n  [🛡️ RAM FRENO] {toplam_kelime} kelimelik kaliteli dilim hazır.")
                break

            if patch_secim == "next" and toplam_kelime >= max(5000, hedef_kelime_limiti // 3):
                break

        for d in yeni_islenen:
            if d not in self.durum["islenen_dosyalar"]:
                self.durum["islenen_dosyalar"].append(d)
        self.durum["toplam_cekilen_kelime"] = self.durum.get("toplam_cekilen_kelime", 0) + toplam_kelime
        self._durum_kaydet()

        if not indirilen:
            log.warning("  [⚠️] Bu turda yeni metin eklenemedi.")
            return mevcut_metin

        yeni_metin = (mevcut_metin + "\n\n" + "\n\n".join(indirilen)).strip()
        log.info(f"\n  [🎉 PATCH AKTARIMI OK] +{toplam_kelime} kaliteli kelime eklendi")
        self.yama_raporu()
        return yeni_metin

    def huggingface_dataset_cek(self, repo_id, mevcut_metin="", hedef_kelime_limiti=40000):
        return self.huggingface_patch_cek(repo_id, "all", mevcut_metin, hedef_kelime_limiti)

    # ==========================================
    # WIKIPEDIA VE DİSK YÖNETİMİ
    # ==========================================
    def wikipedia_cek(self, baslik):
        params = {
            "action": "query", "prop": "extracts", "exintro": False,
            "explaintext": True, "titles": baslik.strip(), "format": "json"
        }
        try:
            yanit = requests.get(self.wiki_api, params=params, headers=self.headers, timeout=10).json()
            for k, v in yanit.get("query", {}).get("pages", {}).items():
                if k != "-1":
                    return self._temizle_metin(v.get("extract", ""))
        except Exception:
            pass
        return None

    def konulari_ogren(self, konular, mevcut_metin=""):
        yeni_metin = mevcut_metin
        for konu in konular:
            makale = self.wikipedia_cek(konu)
            if makale and len(makale) > 150:
                yeni_metin += "\n\n" + makale
        return yeni_metin

    def metni_kaydet(self, metin, dosya_adi="turkce_metin.txt"):
        dosya_yolu = os.path.join(self.proje_kok, dosya_adi)
        with open(dosya_yolu, "w", encoding="utf-8") as f:
            f.write(metin)
        log.info(f"  [💾 KAYIT] Aktif Bilgi Havuzu: {len(metin.split())} kelime")
        return dosya_yolu

    def metni_yukle(self, dosya_adi="turkce_metin.txt"):
        dosya_yolu = os.path.join(self.proje_kok, dosya_adi)
        if os.path.exists(dosya_yolu):
            with open(dosya_yolu, "r", encoding="utf-8") as f:
                return f.read()
        return ""


def main():
    """Komut satırından korpus toplama.

    Örnekler:
        python -m egitim.veri_toplayici                        # sonraki yama
        python -m egitim.veri_toplayici --kelime 60000         # hedef kelime sayısı
        python -m egitim.veri_toplayici --patch all            # tüm dosyalar
        python -m egitim.veri_toplayici --patch 3              # belirli yama no
        python -m egitim.veri_toplayici --repo baska/dataset   # farklı HF kaynağı
        python -m egitim.veri_toplayici --konu "Türkiye"       # Wikipedia makalesi
    """
    kok = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Not: logging burada yerel kurulur; veri toplayıcı torch'a bağımlı olmasın.
    logging.basicConfig(
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        level=logging.INFO, datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Türkçe korpus toplayıcı")
    parser.add_argument("--repo", default=VARSAYILAN_REPO,
                        help=f"Hugging Face veri seti reposu (varsayılan: {VARSAYILAN_REPO})")
    parser.add_argument("--kelime", type=int, default=40000,
                        help="Hedef kelime sayısı (varsayılan 40000)")
    parser.add_argument("--patch", default="next",
                        help="'next' (sonraki yama), 'all' (tümü) veya yama numarası")
    parser.add_argument("--konu", action="append", default=[],
                        help="Wikipedia makalesi başlığı (tekrarlanabilir: --konu a --konu b)")
    parser.add_argument("--cikti", default="turkce_metin.txt", help="Çıktı dosyası adı")
    args = parser.parse_args()

    log_kur()
    vt = OtomatikVeriToplayici(kok, repo_id=args.repo)

    mevcut = vt.metni_yukle(args.cikti)
    if mevcut:
        log.info("Mevcut korpus yüklendi: %s kelime", len(mevcut.split()))

    if args.konu:
        mevcut = vt.konulari_ogren(args.konu, mevcut)
        log.info("Wikipedia konuları eklendi: %s", ", ".join(args.konu))

    yeni = vt.huggingface_patch_cek(args.repo, args.patch, mevcut, args.kelime)
    vt.metni_kaydet(yeni if yeni.strip() else mevcut, args.cikti)


if __name__ == "__main__":
    main()