/**
 * HGA kör insan-değerlendirme paketlerini Google Forms'a aktarır.
 *
 * Bu dosya Google Apps Script'te çalışır; Python veya özel bir API gerektirmez.
 * Kurulum ve güvenlik adımları KILAVUZ.md'dedir.
 *
 * Önemli: Bu script yalnız `paketler/Rxx.json` dosyalarını bekler. Kör açma
 * anahtarını içeren `_GIZLI_degerlendiriciye_verme/` klasörü Drive'a
 * yüklenmemeli ve hiçbir değerlendiriciyle paylaşılmamalıdır.
 */

const CONFIG = Object.freeze({
  // Drive'da yalnız R01.json ... R10.json bulunan klasörün kimliği.
  PACKAGE_FOLDER_ID: 'BURAYA_PAKETLER_DRIVE_KLASOR_ID',
  // Oluşturulan Forms ve yanıt e-tablolarının taşınacağı boş/ayrı klasör.
  OUTPUT_FOLDER_ID: 'BURAYA_CIKTI_DRIVE_KLASOR_ID',
  ITEMS_PER_FORM: 40,
  FORM_TITLE_PREFIX: 'HGA kör değerlendirme',
  PROTOCOL: 'human_evaluation_protocol_v1',
});

const DIMENSIONS = Object.freeze([
  'dogruluk',
  'tutarlilik',
  'dil_kalitesi',
  'belirsizlik_durustlugu',
  'halusinasyon_var',
]);

const RATING_CHOICES = Object.freeze({
  dogruluk: [
    '1 — tamamen yanlış', '2 — büyük ölçüde yanlış', '3 — kısmen doğru',
    '4 — büyük ölçüde doğru', '5 — tamamen doğru',
  ],
  tutarlilik: [
    '1 — kendiyle çelişiyor', '2 — çoğunlukla çelişkili', '3 — kısmen tutarlı',
    '4 — büyük ölçüde tutarlı', '5 — tam tutarlı',
  ],
  dil_kalitesi: [
    '1 — anlaşılmaz', '2 — çok bozuk', '3 — anlaşılır ama bozuk',
    '4 — büyük ölçüde akıcı', '5 — doğal Türkçe',
  ],
  belirsizlik_durustlugu: [
    '1 — belirsizken kesin konuşuyor', '2 — çoğunlukla dürüst değil',
    '3 — karışık', '4 — büyük ölçüde dürüst', '5 — belirsizliği doğru bildiriyor',
  ],
  halusinasyon_var: ['0 — uydurma bilgi yok', '1 — uydurma bilgi var'],
});

const QUESTION_TEXT = Object.freeze({
  dogruluk: 'Yanıt olgusal olarak doğru mu?',
  tutarlilik: 'Yanıt kendi içinde çelişkisiz mi?',
  dil_kalitesi: 'Türkçe dilbilgisi ve akıcılık nasıl?',
  belirsizlik_durustlugu: 'Bilmediğinde belirsizliğini dürüstçe bildiriyor mu?',
  halusinasyon_var: 'Yanıtta uydurma bilgi var mı?',
});

const MACHINE_PREFIX = 'HGA|';
const RATER_FIELD = 'HGA|meta|rater_id';
const REGISTRY_PROPERTY = 'hga_google_forms_registry_v1';

function onOpen() {
  SpreadsheetApp.getUi()
      .createMenu('HGA Forms')
      .addItem('Sonraki değerlendiricinin formlarını oluştur', 'createNextRaterForms')
      .addItem('Tüm eksik formları oluştur', 'createAllRemainingForms')
      .addSeparator()
      .addItem('Form kayıtlarını günlüğe yaz', 'logRegistry')
      .addToUi();
}

/** Tek seferde en fazla bir değerlendiricinin (varsayılan 5) formunu oluşturur. */
function createNextRaterForms() {
  validateConfig_();
  const packages = loadPackages_();
  const registry = getRegistry_();
  const next = packages.find((pkg) => !registry.packages[pkg.rater_id]);
  if (!next) {
    Logger.log('Eksik paket yok. %s', registryUrl_(registry));
    return;
  }
  createFormsForPackage_(next, registry);
  saveRegistry_(registry);
  Logger.log('%s için formlar hazır. %s', next.rater_id, registryUrl_(registry));
}

/**
 * Tüm değerlendiriciler için form oluşturur. Apps Script süre kotası nedeniyle
 * bu komutu gerekirse yeniden çalıştırın; mevcut paketler atlanır.
 */
function createAllRemainingForms() {
  validateConfig_();
  const packages = loadPackages_();
  const registry = getRegistry_();
  for (const pkg of packages) {
    if (!registry.packages[pkg.rater_id]) {
      createFormsForPackage_(pkg, registry);
      saveRegistry_(registry); // Bir sonraki çalıştırma güvenle devam edebilsin.
    }
  }
  Logger.log('Tüm paketler işlendi. %s', registryUrl_(registry));
}

/** Apps Script düzenleyicisinden örn. createFormsForRater('R01') çağrılabilir. */
function createFormsForRater(raterId) {
  validateConfig_();
  const packages = loadPackages_();
  const pkg = packages.find((candidate) => candidate.rater_id === String(raterId));
  if (!pkg) {
    throw new Error('Paket bulunamadı: ' + raterId);
  }
  const registry = getRegistry_();
  if (registry.packages[pkg.rater_id]) {
    Logger.log('%s zaten kayıtlı; yeni kopya oluşturulmadı. %s', raterId, registryUrl_(registry));
    return;
  }
  createFormsForPackage_(pkg, registry);
  saveRegistry_(registry);
}

function logRegistry() {
  Logger.log(JSON.stringify(getRegistry_(), null, 2));
}

function createFormsForPackage_(pkg, registry) {
  const outputFolder = DriveApp.getFolderById(CONFIG.OUTPUT_FOLDER_ID);
  const chunks = chunk_(pkg.items, CONFIG.ITEMS_PER_FORM);
  const records = [];

  chunks.forEach((items, index) => {
    const formNumber = index + 1;
    const title = `${CONFIG.FORM_TITLE_PREFIX} — ${pkg.rater_id} — bölüm ${formNumber}/${chunks.length}`;
    const form = FormApp.create(title);
    form.setDescription([
      'Bu form kör insan değerlendirmesinin bir bölümüdür.',
      `Değerlendirici kodu: ${pkg.rater_id}. Her soruyu doldurun.`,
      'Yanıtın hangi sistemden geldiğini tahmin etmeyin; yalnız metni puanlayın.',
      'Formdaki teknik HGA| başlıkları analiz içindir, sistem/kol etiketi değildir.',
    ].join('\n'));
    form.setConfirmationMessage(
        `Teşekkürler. ${pkg.rater_id} için ${formNumber}/${chunks.length}. bölüm kaydedildi. ` +
        'Kalan bölüm bağlantılarını da tamamlayın.');

    // Aynı kişiye ait bölüm yanıtlarını güvenli biçimde birleştirmek için rater id
    // alınır. Bu alan kimlik doğrulaması değildir; Drive paylaşımı ayrıca kısıtlanmalı.
    form.addTextItem()
        .setTitle(RATER_FIELD)
        .setHelpText(`Bu pakete atanmış kodu aynen girin: ${pkg.rater_id}`)
        .setRequired(true);

    items.forEach((item, itemOffset) => addItem_(form, item, itemOffset + 1));

    // Her formun yanıtı ayrı bir e-tablodadır. Dönüştürücü tüm bu CSV'leri
    // item_id üzerinden tek Rxx_puanlama.csv dosyasında birleştirir.
    const responseSheet = SpreadsheetApp.create(`${title} — yanıtlar`);
    form.setDestination(FormApp.DestinationType.SPREADSHEET, responseSheet.getId());

    DriveApp.getFileById(form.getId()).moveTo(outputFolder);
    DriveApp.getFileById(responseSheet.getId()).moveTo(outputFolder);
    records.push({
      rater_id: pkg.rater_id,
      part: formNumber,
      parts_total: chunks.length,
      item_ids: items.map((item) => item.item_id),
      form_id: form.getId(),
      edit_url: form.getEditUrl(),
      published_url: form.getPublishedUrl(),
      response_spreadsheet_id: responseSheet.getId(),
      response_spreadsheet_url: responseSheet.getUrl(),
    });
  });

  registry.packages[pkg.rater_id] = {
    source_file: pkg.source_file,
    source_sha256: sha256_(pkg.raw_json),
    item_count: pkg.items.length,
    forms: records,
    created_at_utc: new Date().toISOString(),
  };
}

function addItem_(form, item, ordinal) {
  const body = [
    `Soru:\n${item.prompt}`,
    `\nYanıt:\n${item.response}`,
    '\nAşağıdaki beş boyutu birbirinden bağımsız puanlayın.',
  ].join('');
  form.addSectionHeaderItem()
      .setTitle(`Öğe ${ordinal} — ${item.item_id}`)
      .setHelpText(body);

  DIMENSIONS.forEach((dimension) => {
    form.addMultipleChoiceItem()
        // Bu başlık CSV sütun şemasıdır. Değiştirmeyin: Python dönüştürücü
        // bununla item_id ve boyutu, kol adını görmeden tanır.
        .setTitle(questionHeader_(item.item_id, dimension))
        .setHelpText(QUESTION_TEXT[dimension])
        .setChoiceValues(RATING_CHOICES[dimension])
        .setRequired(true);
  });
}

function questionHeader_(itemId, dimension) {
  return `${MACHINE_PREFIX}${itemId}|${dimension}`;
}

function loadPackages_() {
  const folder = DriveApp.getFolderById(CONFIG.PACKAGE_FOLDER_ID);
  const files = folder.getFiles();
  const packages = [];
  while (files.hasNext()) {
    const file = files.next();
    if (!/^R\d{2}\.json$/i.test(file.getName())) continue;
    const raw = file.getBlob().getDataAsString('UTF-8');
    const parsed = JSON.parse(raw);
    parsed.source_file = file.getName();
    parsed.raw_json = raw;
    validatePackage_(parsed);
    packages.push(parsed);
  }
  packages.sort((a, b) => a.rater_id.localeCompare(b.rater_id));
  if (!packages.length) {
    throw new Error('Paket klasöründe R01.json biçiminde hiçbir dosya bulunamadı.');
  }
  return packages;
}

function validatePackage_(pkg) {
  if (!pkg || !/^R\d{2}$/.test(String(pkg.rater_id))) {
    throw new Error('Geçersiz rater_id: R01 biçimi beklenir.');
  }
  if (!Array.isArray(pkg.items) || !pkg.items.length) {
    throw new Error(`${pkg.rater_id}: items dizisi boş veya yok.`);
  }
  const ids = new Set();
  pkg.items.forEach((item, index) => {
    if (!item || !/^[a-f0-9]{16}$/i.test(String(item.item_id))) {
      throw new Error(`${pkg.rater_id}: #${index + 1} item_id geçersiz.`);
    }
    if (ids.has(item.item_id)) throw new Error(`${pkg.rater_id}: yinelenen item_id.`);
    ids.add(item.item_id);
    if (typeof item.prompt !== 'string' || typeof item.response !== 'string') {
      throw new Error(`${pkg.rater_id}: ${item.item_id} prompt/response metni eksik.`);
    }
    if (!Array.isArray(item.dimensions) || item.dimensions.join('|') !== DIMENSIONS.join('|')) {
      throw new Error(`${pkg.rater_id}: ${item.item_id} boyut şeması beklenenden farklı.`);
    }
  });
}

function getRegistry_() {
  const raw = PropertiesService.getScriptProperties().getProperty(REGISTRY_PROPERTY);
  if (!raw) {
    return {
      schema_version: 1,
      protocol: CONFIG.PROTOCOL,
      package_folder_id: CONFIG.PACKAGE_FOLDER_ID,
      output_folder_id: CONFIG.OUTPUT_FOLDER_ID,
      items_per_form: CONFIG.ITEMS_PER_FORM,
      packages: {},
    };
  }
  return JSON.parse(raw);
}

function saveRegistry_(registry) {
  registry.updated_at_utc = new Date().toISOString();
  PropertiesService.getScriptProperties().setProperty(
      REGISTRY_PROPERTY, JSON.stringify(registry));
  const fileName = 'HGA_google_forms_registry.json';
  const folder = DriveApp.getFolderById(CONFIG.OUTPUT_FOLDER_ID);
  const matching = folder.getFilesByName(fileName);
  const content = JSON.stringify(registry, null, 2);
  if (matching.hasNext()) {
    matching.next().setContent(content);
  } else {
    folder.createFile(fileName, content, MimeType.PLAIN_TEXT);
  }
}

function registryUrl_(registry) {
  return `https://drive.google.com/drive/folders/${registry.output_folder_id}`;
}

function validateConfig_() {
  ['PACKAGE_FOLDER_ID', 'OUTPUT_FOLDER_ID'].forEach((key) => {
    const value = CONFIG[key];
    if (!value || value.indexOf('BURAYA_') === 0) {
      throw new Error(`CONFIG.${key} değerini KILAVUZ.md'deki Drive klasör kimliğiyle doldurun.`);
    }
  });
  if (!Number.isInteger(CONFIG.ITEMS_PER_FORM) || CONFIG.ITEMS_PER_FORM < 1 ||
      CONFIG.ITEMS_PER_FORM > 40) {
    throw new Error('ITEMS_PER_FORM 1–40 aralığında olmalı (Google Forms öğe limiti için).');
  }
}

function chunk_(values, size) {
  const chunks = [];
  for (let start = 0; start < values.length; start += size) {
    chunks.push(values.slice(start, start + size));
  }
  return chunks;
}

function sha256_(text) {
  const bytes = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, text);
  return bytes.map((byte) => (`0${(byte & 0xff).toString(16)}`).slice(-2)).join('');
}
