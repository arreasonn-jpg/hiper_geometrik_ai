# -*- coding: utf-8 -*-
"""
HGA Experience Engine — Komut Satırı Arayüzü
=============================================
Kullanım:
    python -m hga bilgi                  # bilgi tabanı demosu (Ali/Ata/Araba/Gökyüzü)
    python -m hga gercek-veri            # gerçek veri → temsil → deneyim → doğrulama
    python -m hga benchmark              # kontrollü benchmark (metrik tablosu)
    python -m hga golden-benchmark       # elle sabit golden set + FAR/FRR/leakage
    python -m hga memory-benchmark       # collision/interference/retrieval stres testi
    python -m hga kronecker-benchmark    # eşit-parametre Kronecker/rank-1 kıyası
    python -m hga self-learning-benchmark # K₀→Kₙ + collapse failure injection
    python -m hga research-benchmark     # birleşik, 5-seed JSON/MD/HTML araştırma karnesi
    python -m hga milestone              # K₀→Kₙ milestone tablosu (versiyon + defter)
    python -m hga kapasite               # P / C_I^UB / C_M^UB / C_E / C_V kapasite çerçevesi
    python -m hga bilgi-surum            # bilgi sürümleme + rollback demosu
    python -m hga defter                 # immutable experience ledger demosu
    python -m hga memory-interference    # kasıtlı çakışma + sabit/dinamik KV kıyası
    python -m hga paradigma              # neural vs symbolic vs hybrid (Faz 21)
    python -m hga olcekli-golden         # 100/1K/10K golden benchmark (Faz 3/6)
    python -m hga kronecker-rank         # effective rank + zincir çöküşü (Faz 19/20)
    python -m hga epistemik              # KNOWN/UNKNOWN/UNCERTAIN/CONFLICT/FALSE (P0-007)
    python -m hga verim                  # NY / UEY / GY / VID verim ayrıştırması (P1-005)
    python -m hga koken                  # provenance denetimi (Faz 27/28)
    python -m hga oncelik                # Priority(E) ağırlık ablasyonu (Faz 25)
    python -m hga oncelik-zincir         # Priority(E) nedensel zincir ablasyonu (P0-1)
    python -m hga operator-baseline      # Kronecker vs gerçek Dense ailesi (P0-2)
    python -m hga signature              # HGA Signature Benchmark v1 (P0-4)
    python -m hga signature-gate         # Signature release gate (PASS/BLOCKED)
    python -m hga verifier-adversarial   # verifier adversarial suite v2
    python -m hga verifier-ensemble      # üç üyeli verifier ensemble
    python -m hga semantik               # Türkçe semantik çıkarım benchmarkı (P0-5)
    python -m hga genelleme-v2           # ham metinden keşif + C_G v2 (P0-6)
    python -m hga cikarim-derinligi      # C_R / C_RD çıkarım derinliği (P0-7)
    python -m hga twt-sonuc              # TWT gerçek sonuç tablosu + FLOPs (P0-3)
    python -m hga english-ewt            # UD English EWT + Transformer/BERT/GPT-style baseline'ları
    python -m hga english-scaling        # HGA English EWT parameter scaling probe (scaling law değildir)
    python -m hga foundation-analysis    # CKPT-001 rank/gradient/hash teorik tanı grid'i
    python -m hga reproduce-all          # manifestli CKPT-001 foundation reproduction bundle
    python -m hga turkce-lm              # gerçek Türkçe LM: tr_corpus_v1 1.11M kelime (P1)
    python -m hga uzun-baglam            # uzun bağlam LM; 512/1024 için --long-context-profile smoke_1024
    python -m hga bellek-hiyerarsi       # hot/warm/cold/archive bellek (P0-8)
    python -m hga bellek-streaming       # 100M-ready streaming/checkpoint smoke harness
    python -m hga cok-ortam              # 5 ortam + verifier izolasyonu (P1)
    python -m hga ogrenme-transfer       # cross-domain self-learning transfer negatif-kontrol
    python -m hga ogrenme-olcek          # self-learning 100→1000→3000 cycle (P1)
    python -m hga tohum-istatistik       # 20 tohum + CI/etki/permütasyon (P1)
    python -m hga insan-degerlendirme    # protokol + Krippendorff alfa (P1)
    python -m hga insan-import           # insan puanı CSV import + α + kol agregasyonu
    python -m hga korpus-genisletme      # Türkçe korpus genişletme pipeline/release gate
    python -m hga derinlik-teshis        # derinlik çöküşü kök neden
    python -m hga oncelik-optimizasyon   # ağırlık araması + held-out
    python -m hga muhendislik            # CI / paketleme / test sözleşmesi
    python -m hga yeniden-uretilebilirlik # manifest / determinizm / tohum
    python -m hga championship-benchmark # tüm P0+P1 + OTOMATİK araştırma karnesi
    python -m hga dogrulama              # kapalı doğrulama hattı (false accept 24→0)
    python -m hga halusinasyon           # factual consistency / hallucination metriği
    python -m hga sweep                  # n/K/context kapasite taraması
    python -m hga tokenizer              # mini Türkçe tokenizer benchmark
    python -m hga perplexity [dosya]     # mini/tiny Türkçe perplexity smoke
    python -m hga checkpoint-rapor ckpt.pt # checkpoint/model şekil uyumluluğu
    python -m hga benchmark-rapor --out rapor.json --markdown rapor.md
    python -m hga veri-kalite [dosya]    # veri kalite filtresi raporu
    python -m hga veri-canli-smoke       # küçük canlı Türkçe veri hattı smoke
    python -m hga manifest dosya.txt     # SHA-256 veri manifesti
    python -m hga observability          # bellek/deneyim/geometri gözlem demosu/paneli
    python -m hga ozet [bilgi.json]      # bilgi tabanı özeti (opsiyonel yükleme)
"""
import argparse
import sys

from hga.cli.benchmark import _championship, _research_benchmark
from hga.cli.common import KOK, MIMARI
from hga.cli.demos import (
    _benchmark,
    _bilgi_demo,
    _defter,
    _dogrulama,
    _gercek_veri,
    _graf_demo,
    _halusinasyon,
    _kesif_demo,
    _observability_demo,
    _ozet,
    _sweep,
)
from hga.cli.depth import _cikarim_derinligi
from hga.cli.english import _english_ewt, _english_scaling
from hga.cli.epistemik import _epistemik
from hga.cli.eval import (
    _benchmark_rapor,
    _bilgi_surum,
    _cok_adimli,
    _derinlik_teshis,
    _golden_benchmark,
    _insan_import,
    _kapasite,
    _korpus_genisletme,
    _kronecker_benchmark,
    _memory_benchmark,
    _memory_interference,
    _muhendislik,
    _olcekli_golden,
    _oncelik_optimizasyon,
    _oncelik_zincir,
    _paradigma,
    _priority,
    _reproduce_all,
    _turkce_lm,
    _twt_sonuc,
    _verim,
    _yeniden_uretilebilirlik,
)
from hga.cli.foundation import _foundation_analysis
from hga.cli.generalization import _genelleme_v2
from hga.cli.human_eval import _insan_degerlendirme
from hga.cli.kronecker_rank import _kronecker_rank
from hga.cli.learning import _ogrenme_olcek, _ogrenme_transfer, _self_learning_benchmark
from hga.cli.long_context import _uzun_baglam
from hga.cli.memory import _bellek_hiyerarsi, _bellek_streaming
from hga.cli.milestone import _milestone
from hga.cli.multi_env import _cok_ortam
from hga.cli.operator_baseline import _operator_baseline
from hga.cli.provenance import _provenance
from hga.cli.seed_statistics import _tohum_istatistik
from hga.cli.semantic import _semantik
from hga.cli.signature import _signature, _signature_gate
from hga.cli.tokenizer import (
    _checkpoint_rapor,
    _manifest,
    _perplexity_benchmark,
    _tokenizer_benchmark,
    _veri_canli_smoke,
    _veri_kalite,
)
from hga.cli.verifier import _verifier_adversarial, _verifier_ensemble

for _p in [KOK, MIMARI]:
    if _p not in sys.path:
        sys.path.insert(0, _p)











































































































































def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m hga",
                                description="HGA Experience Engine CLI")
    p.add_argument("komut", choices=["bilgi", "gercek-veri", "benchmark",
                                     "dogrulama", "halusinasyon", "sweep",
                                     "tokenizer", "perplexity", "checkpoint-rapor",
                                     "benchmark-rapor", "veri-kalite", "veri-canli-smoke",
                                     "manifest", "observability", "ozet",
                                     "graf", "kesif", "golden-benchmark",
                                     "memory-benchmark", "kronecker-benchmark",
                                     "self-learning-benchmark", "research-benchmark", "milestone",
                                     "kapasite", "bilgi-surum", "defter",
                                     "memory-interference", "paradigma",
                                     "olcekli-golden", "kronecker-rank",
                                     "epistemik", "verim", "cok-adimli",
                                     "koken", "oncelik",
                                     "oncelik-zincir", "operator-baseline",
                                     "cikarim-derinligi", "signature",
                                     "signature-gate",
                                     "verifier-adversarial",
                                     "verifier-ensemble",
                                     "semantik", "genelleme-v2",
                                     "twt-sonuc", "english-ewt", "english-scaling", "foundation-analysis",
                                     "reproduce-all", "turkce-lm",
                                     "uzun-baglam",
                                     "bellek-hiyerarsi",
                                     "bellek-streaming",
                                     "cok-ortam", "ogrenme-transfer",
                                     "ogrenme-olcek",
                                     "tohum-istatistik",
                                     "insan-degerlendirme",
                                     "insan-import", "korpus-genisletme",
                                     "derinlik-teshis",
                                     "oncelik-optimizasyon",
                                     "muhendislik",
                                     "yeniden-uretilebilirlik",
                                     "championship-benchmark"])
    p.add_argument("yol", nargs="?", default=None,
                   help="dosya yolu: ozet/veri-kalite/manifest/perplexity/checkpoint-rapor")
    p.add_argument("--config", default=None,
                   help="checkpoint-rapor için model/eğitim YAML config yolu")
    p.add_argument("--checkpoint", default=None,
                   help="perplexity için strict yüklenecek model checkpoint yolu")
    p.add_argument("--tokenizer-yol", default=None,
                   help="perplexity için BPE sözlük yolu (varsayılan: bpe_sozluk.json)")
    p.add_argument("--tiny", action="store_true",
                   help="perplexity smoke için küçük model kullan (varsayılan; --full-config bunu kapatır)")
    p.add_argument("--full-config", action="store_true",
                   help="perplexity için tiny yerine model_config.yaml boyutlarını kullan")
    p.add_argument("--batch", type=int, default=64,
                   help="perplexity batch boyutu")
    p.add_argument("--n", type=int, default=None,
                   help="perplexity tiny/full model n override")
    p.add_argument("--katman", type=int, default=None,
                   help="perplexity K override")
    p.add_argument("--baglam", type=int, default=None,
                   help="perplexity bağlam override")
    p.add_argument("--vocab", type=int, default=None,
                   help="perplexity/checkpoint model vocab override")
    p.add_argument("--emb", type=int, default=None,
                   help="checkpoint-rapor için emb_dim override")
    p.add_argument("--heads", type=int, default=None,
                   help="checkpoint-rapor için attention head override")
    p.add_argument("--seyrek-satir", type=int, default=None,
                   help="checkpoint-rapor için seyrek tablo satır override")
    p.add_argument("--seyrek-boyut", type=int, default=None,
                   help="checkpoint-rapor için seyrek vektör boyutu override")
    p.add_argument("--no-strict", action="store_true",
                   help="checkpoint-rapor için eksik/fazla anahtarı hata sayma")
    p.add_argument("--out", default=None,
                   help="rapor üreten komutlar için JSON çıktı yolu")
    p.add_argument("--markdown", default=None,
                   help="rapor üreten komutlar için Markdown çıktı yolu")
    p.add_argument("--html", default=None,
                   help="observability/research-benchmark için HTML çıktı yolu")
    p.add_argument("--konular", default=None,
                   help="veri-canli-smoke için virgüllü Wikipedia konu listesi")
    p.add_argument("--cikis", default=None,
                   help="veri-canli-smoke için temizlenmiş küçük corpus çıktı yolu")
    p.add_argument("--kontrollu", action="store_true",
                   help="veri-canli-smoke için ağsız/deterministik fetcher kullan")
    p.add_argument("--seeds", default=None,
                   help="çoklu-seed benchmarklar için virgüllü liste (örn. 1,2,3,4,5)")
    p.add_argument("--profile", choices=["smoke", "full"], default="smoke",
                   help="research-benchmark çalışma profili")
    p.add_argument("--sections", default=None,
                   help="research-benchmark bölüm filtresi (virgüllü; varsayılan: tümü)")
    p.add_argument("--strict", action="store_true",
                   help="research-benchmark bölüm atlanır/hata verirse non-zero çık")
    p.add_argument("--n-values", default="4,8,16",
                   help="kronecker-rank için n değerleri (virgüllü)")
    p.add_argument("--k-values", default="1,2,4",
                   help="kronecker-rank için K değerleri (virgüllü)")
    p.add_argument("--force", action="store_true",
                   help="kronecker-rank: güvensiz n/K konfigürasyonlarını "
                        "zorla çalıştır (disk/RAM dolabilir)")
    p.add_argument("--sizes", default="100,1000,10000",
                   help="olcekli-golden test seti boyutları (virgüllü)")
    p.add_argument("--hard", action="store_true",
                   help="olcekli-golden: sınır vakalarını (eşik/öncelik/kaynak) ekle")
    p.add_argument("--epochs", type=int, default=60,
                   help="paradigma nöral kolu eğitim epoch sayısı")
    p.add_argument("--experiment-root", default="experiments",
                   help="EXP-NNNN çalışma dizinlerinin kökü")
    p.add_argument("--artifact-root", default="artifacts/reproduce-all",
                   help="reproduce-all için üst artefakt dizini")
    p.add_argument("--scales", default="1000,10000,100000",
                   help="memory-benchmark context ölçekleri")
    p.add_argument("--slots", type=int, default=65536,
                   help="memory-benchmark için tablo başına slot sayısı")
    p.add_argument("--tables", default="1,2",
                   help="memory-benchmark tablo sayıları (1,2 veya ikisi)")
    p.add_argument("--hops", default=None,
                   help="cok-adimli: zincir derinlikleri (örn 1,2,3,4,5)")
    p.add_argument("--distractors", default=None,
                   help="cok-adimli: araya giren dolgu olgu sayıları (örn 0,16,64,256)")
    p.add_argument("--active-dynamic", action="store_true",
                   help="memory-benchmark: aynı streamde aktif bounded Dynamic KV'yi de kır")
    p.add_argument("--audit-samples", type=int, default=256,
                   help="aktif memory stress checkpoint exact audit örnek sayısı")
    p.add_argument("--steps", type=int, default=100,
                   help="kronecker-benchmark optimizasyon adımı")
    p.add_argument("--lm-steps", type=int, default=None,
                   help="turkce-lm eğitim adımı override (yalnız smoke/CI "
                        "hızlandırması; hash'e dahildir)")
    p.add_argument("--device", default="cpu",
                   help="kronecker-benchmark torch cihazı (cpu/cuda)")
    p.add_argument("--cycles", type=int, default=100,
                   help="self-learning/collapse döngü sayısı")
    p.add_argument("--initial-facts", type=int, default=100,
                   help="self-learning başlangıç doğrulanmış bilgi sayısı")
    p.add_argument("--operands-max", type=int, default=31,
                   help="self-learning aritmetik domain üst operandı")
    p.add_argument("--negatives-per-fact", type=int, default=3,
                   help="her doğru frontier olgusu başına yanlış aday")
    p.add_argument("--memory-slots", type=int, default=4096,
                   help="self-learning/milestone deney belleği slot sayısı")
    p.add_argument("--checkpoints", default=None,
                   help="milestone tablosu kontrol noktaları (örn. 0,10,50,100)")
    p.add_argument("--forced-collisions", type=int, default=25,
                   help="memory-interference için kasıtlı çakışma sayısı")
    p.add_argument("--signature-profile", choices=["smoke", "standard"],
                   default="smoke",
                   help="signature / championship-benchmark ölçek profili")
    p.add_argument("--depth-profile", choices=["smoke", "standard", "deep"],
                   default="standard",
                   help="cikarim-derinligi tarama profili")
    p.add_argument("--long-context-profile", choices=["smoke", "smoke_1024", "full"],
                   default="smoke",
                   help="uzun-baglam profili; smoke_1024 512/1024 yolunu koşar")
    p.add_argument("--reliability-threshold", type=float, default=1.0,
                   help="C_R/C_RD güvenilirlik eşiği (0,1]")
    p.add_argument("--skip-torch", action="store_true",
                   help="championship-benchmark: nöral bölümleri atla")
    p.add_argument("--ledger", default=None,
                   help="milestone için immutable experience ledger JSONL yolu")
    p.add_argument("--target-records", type=int, default=100_000_000,
                   help="bellek-streaming için hedef kayıt sayısı")
    p.add_argument("--sample-records", type=int, default=10_000,
                   help="bellek-streaming smoke koşusunda gerçek yazılacak kayıt")
    p.add_argument("--shard-records", type=int, default=1_000_000,
                   help="bellek-streaming planında shard başına kayıt")
    p.add_argument("--checkpoint-interval", type=int, default=1_000,
                   help="bellek-streaming checkpoint aralığı")
    p.add_argument("--source-examples", type=int, default=120,
                   help="ogrenme-transfer için kaynak ortam örnek sayısı")
    p.add_argument("--target-support-examples", type=int, default=40,
                   help="ogrenme-transfer için hedef destek örneği")
    p.add_argument("--target-unseen-examples", type=int, default=40,
                   help="ogrenme-transfer için hedef unseen eval örneği")
    p.add_argument("--package-dir", default="insan_degerlendirme_paketleri",
                   help="insan-import/degerlendirme paket dizini")
    p.add_argument("--candidate-dir", default=None,
                   help="korpus-genisletme için aday JSONL dizini")
    p.add_argument("--no-smoke-candidates", action="store_true",
                   help="korpus-genisletme: aday dizini yoksa gömülü smoke aday kullanma")
    p.add_argument("--target-words", type=int, default=2_000_000,
                   help="korpus-genisletme release hedef kelime sayısı")
    args = p.parse_args(argv)
    {"bilgi": _bilgi_demo, "gercek-veri": _gercek_veri,
     "benchmark": _benchmark, "dogrulama": _dogrulama,
     "halusinasyon": _halusinasyon, "sweep": _sweep,
     "tokenizer": _tokenizer_benchmark,
     "graf": _graf_demo, "kesif": _kesif_demo,
     "perplexity": lambda: _perplexity_benchmark(
         args.yol, checkpoint=args.checkpoint,
         tokenizer_yol=args.tokenizer_yol,
         tiny=(args.tiny or not args.full_config), batch_size=args.batch,
         n=args.n, katman=args.katman, baglam=args.baglam, vocab=args.vocab),
     "checkpoint-rapor": lambda: _checkpoint_rapor(
         args.yol, config_yol=args.config, strict=not args.no_strict,
         n=args.n, katman=args.katman, baglam=args.baglam, vocab=args.vocab,
         emb=args.emb, heads=args.heads, seyrek_satir=args.seyrek_satir,
         seyrek_boyut=args.seyrek_boyut),
     "benchmark-rapor": lambda: _benchmark_rapor(
         out=args.out, markdown=args.markdown, checkpoint=args.checkpoint,
         tokenizer_yol=args.tokenizer_yol, n=args.n, katman=args.katman,
         baglam=args.baglam, vocab=args.vocab),
     "veri-kalite": lambda: _veri_kalite(args.yol),
     "veri-canli-smoke": lambda: _veri_canli_smoke(
         konular=args.konular, cikis=args.cikis, out=args.out,
         kontrollu=args.kontrollu),
     "manifest": lambda: _manifest(args.yol),
     "golden-benchmark": lambda: _golden_benchmark(
         args.out, seeds=args.seeds, experiment_root=args.experiment_root),
     "memory-benchmark": lambda: _memory_benchmark(
         args.scales, args.slots, args.tables, args.seeds or "42",
         args.experiment_root, out=args.out, markdown=args.markdown,
         active_dynamic=args.active_dynamic, audit_samples=args.audit_samples),
     "kronecker-benchmark": lambda: _kronecker_benchmark(
         int(args.n or 16), args.steps, args.batch, args.seeds or "1,2,3,4,5",
         args.device, args.experiment_root, out=args.out),
     "self-learning-benchmark": lambda: _self_learning_benchmark(
         args.cycles, args.batch, args.initial_facts, args.operands_max,
         args.negatives_per_fact, args.memory_slots, args.seeds or "42",
         args.experiment_root, out=args.out),
     "research-benchmark": lambda: _research_benchmark(
         args.profile, args.seeds or "1,2,3,4,5", args.experiment_root,
         sections=args.sections, out=args.out, markdown=args.markdown,
         html_yol=args.html, strict=args.strict),
     "milestone": lambda: _milestone(
         args.cycles, args.batch, args.initial_facts, args.operands_max,
         args.negatives_per_fact, args.memory_slots, args.seeds or "42",
         args.experiment_root, out=args.out, markdown=args.markdown,
         ledger=args.ledger, checkpoints=args.checkpoints),
     "memory-interference": lambda: _memory_interference(
         args.forced_collisions, args.slots, 1,
         args.scales if args.scales != "1000,10000,100000" else "100,1000,10000,100000",
         out=args.out, markdown=args.markdown),
     "paradigma": lambda: _paradigma(args.seeds or "1,2,3,4,5", out=args.out,
                                     markdown=args.markdown,
                                     epochs=args.epochs),
     "olcekli-golden": lambda: _olcekli_golden(
         args.sizes, args.seeds or "1,2,3,4,5", hard=args.hard,
         out=args.out, markdown=args.markdown),
     "kronecker-rank": lambda: _kronecker_rank(
         args.n_values, args.k_values, seed=1,
         out=args.out, markdown=args.markdown, force=args.force),
     "epistemik": lambda: _epistemik(args.seeds, out=args.out,
                                     markdown=args.markdown),
     "cok-adimli": lambda: _cok_adimli(args.seeds, hops=args.hops,
                                       distractors=args.distractors,
                                       slots=args.memory_slots,
                                       out=args.out, markdown=args.markdown),
     "verim": lambda: _verim(args.cycles, args.batch, args.initial_facts,
                             args.operands_max, args.negatives_per_fact,
                             args.seeds, out=args.out, markdown=args.markdown),
     "koken": lambda: _provenance(out=args.out, markdown=args.markdown),
     "oncelik": lambda: _priority(k=10, out=args.out, markdown=args.markdown),
     "oncelik-zincir": lambda: _oncelik_zincir(
         k=10, seeds=args.seeds, out=args.out, markdown=args.markdown),
     "operator-baseline": lambda: _operator_baseline(
         n=int(args.n or 8), steps=args.steps, seeds=args.seeds,
         device=args.device, out=args.out, markdown=args.markdown),
     "cikarim-derinligi": lambda: _cikarim_derinligi(
         profile=args.depth_profile, seeds=args.seeds,
         threshold=args.reliability_threshold, out=args.out,
         markdown=args.markdown),
     "signature": lambda: _signature(
         profile=args.signature_profile, seeds=args.seeds, out=args.out,
         markdown=args.markdown),
     "signature-gate": lambda: _signature_gate(
         yol=args.yol, profile=args.signature_profile, seeds=args.seeds,
         out=args.out, markdown=args.markdown),
     "verifier-adversarial": lambda: _verifier_adversarial(
         out=args.out, markdown=args.markdown),
     "verifier-ensemble": lambda: _verifier_ensemble(
         out=args.out, markdown=args.markdown),
     "semantik": lambda: _semantik(out=args.out, markdown=args.markdown),
     "genelleme-v2": lambda: _genelleme_v2(out=args.out, markdown=args.markdown),
     "twt-sonuc": lambda: _twt_sonuc(
         seeds=args.seeds, profile=args.signature_profile, out=args.out,
         markdown=args.markdown),
     "english-ewt": lambda: _english_ewt(
         seeds=args.seeds, profile=args.signature_profile, out=args.out,
         markdown=args.markdown),
     "english-scaling": lambda: _english_scaling(
         seeds=args.seeds, profile=args.signature_profile, out=args.out,
         markdown=args.markdown),
     "foundation-analysis": lambda: _foundation_analysis(
         n_values=args.n_values, k_values=args.k_values,
         seed=(int(args.seeds.split(",")[0]) if args.seeds else 1),
         out=args.out, markdown=args.markdown),
     "reproduce-all": lambda: _reproduce_all(
         artifact_root=args.artifact_root, seeds=args.seeds, profile=args.profile),
     "turkce-lm": lambda: _turkce_lm(
         seeds=args.seeds, profile=args.signature_profile,
         steps=args.lm_steps, out=args.out, markdown=args.markdown),
     "uzun-baglam": lambda: _uzun_baglam(
         profile=args.long_context_profile, seeds=args.seeds,
         out=args.out, markdown=args.markdown),
     "bellek-hiyerarsi": lambda: _bellek_hiyerarsi(
         profile=args.depth_profile, seeds=args.seeds, out=args.out,
         markdown=args.markdown),
     "bellek-streaming": lambda: _bellek_streaming(
         target_records=args.target_records,
         sample_records=args.sample_records,
         shard_records=args.shard_records,
         checkpoint_interval=args.checkpoint_interval,
         seeds=args.seeds, out=args.out, markdown=args.markdown),
     "cok-ortam": lambda: _cok_ortam(
         seeds=args.seeds, out=args.out, markdown=args.markdown),
     "ogrenme-transfer": lambda: _ogrenme_transfer(
         seeds=args.seeds, source_examples=args.source_examples,
         target_support_examples=args.target_support_examples,
         target_unseen_examples=args.target_unseen_examples,
         out=args.out, markdown=args.markdown),
     "ogrenme-olcek": lambda: _ogrenme_olcek(
         profile=args.signature_profile, seeds=args.seeds, out=args.out,
         markdown=args.markdown),
     "tohum-istatistik": lambda: _tohum_istatistik(
         profile=args.signature_profile, seeds=args.seeds, out=args.out,
         markdown=args.markdown),
     "insan-degerlendirme": lambda: _insan_degerlendirme(
         package_dir=args.package_dir, out=args.out, markdown=args.markdown),
     "insan-import": lambda: _insan_import(
         package_dir=args.package_dir, out=args.out, markdown=args.markdown),
     "korpus-genisletme": lambda: _korpus_genisletme(
         candidate_dir=args.candidate_dir, target_words=args.target_words,
         use_smoke_candidates=not args.no_smoke_candidates,
         out=args.out, markdown=args.markdown),
     "derinlik-teshis": lambda: _derinlik_teshis(
         profile=args.depth_profile, out=args.out, markdown=args.markdown),
     "oncelik-optimizasyon": lambda: _oncelik_optimizasyon(
         profile=args.signature_profile, out=args.out,
         markdown=args.markdown),
     "muhendislik": lambda: _muhendislik(out=args.out, markdown=args.markdown),
     "yeniden-uretilebilirlik": lambda: _yeniden_uretilebilirlik(
         out=args.out, markdown=args.markdown),
     "championship-benchmark": lambda: _championship(
         profile=args.signature_profile, seeds=args.seeds, out=args.out,
         markdown=args.markdown, skip_torch=args.skip_torch),
     "kapasite": lambda: _kapasite(args.operands_max, out=args.out),
     "bilgi-surum": lambda: _bilgi_surum(out=args.out),
     "defter": lambda: _defter(out=args.out),
     "observability": lambda: _observability_demo(
         out=args.out, markdown=args.markdown, html_yol=args.html),
     "ozet": lambda: _ozet(args.yol)}[args.komut]()


if __name__ == "__main__":
    main()
