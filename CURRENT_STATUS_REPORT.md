# Системийн одоогийн бодит байдал — тайланд ашиглах материал

Энэ файл нь тайлан/бакалаврын ажилд оруулах мэдээллийг нэгтгэсэн.

---

## I. Системийн одоогийн бодит байдал

### Production дээр одоо юу ажиллаж байна вэ?

- **Одоо:** Rule-based + ML-ийг **нэгтгэсэн hybrid** prediction ажиллаж байна.
- **ML model файл** (`dzud_risk_model_advanced.pkl`, `scaler_advanced.pkl`) **байгаа бол** API-д ашиглагдана:
  - Тухайн байршил/сар дээр 24 feature бэлтгэж, модельд `predict_proba` дуудана.
  - Гарсан зудын магадлал (0–1)-ийг rule-based эрсдэлийн оноо (0–100)-тай **50/50** харьцаагаар нэгтгэн эцсийн эрсдэл гаргана.
  - Хариунд `risk.risk_source: 'hybrid'`, `risk.ml_probability` (магадлал) орно.
- **Model файл байхгүй** эсвэл feature бэлтгэх өгөгдөл дутуу бол зөвхөн **rule-based** ажиллана (`risk_source: 'rule_based'`).

**Дүгнэлт:** Зөвхөн rule-based гэж үлдсэн биш; ML файл байгаа үед **hybrid** ажилладаг.

---

## II. ML загварууд — бодит тоо, тохиргоо

### Ямар алгоритмууд туршсан бэ?

Дараах **3** classifier-ийг `train_model_advanced.py` дотор туршиж, **accuracy-гаар хамгийн сайн** нэгийг сонгон хадгална:

| Алгоритм | Тохиргоо (товч) |
|----------|------------------|
| **Logistic Regression** | max_iter=2000, class_weight='balanced', StandardScaler-тэй |
| **Random Forest** | n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, class_weight='balanced' |
| **Gradient Boosting** | n_estimators=200, max_depth=7, learning_rate=0.05 |

Одоогийн хадгалсан загвар: **Gradient Boosting** (metadata-аас).

### Accuracy, AUC, F1 бодитоор хэд гарсан бэ?

- **Accuracy:** Метадатад бүртгэгдсэн — **~0.9804** (98.04%). Энэ нь **сонгогдсон best model**-ийн test set дээрх accuracy.
- **AUC-ROC:** Сургалтын скрипт ажиллахад `roc_auc_score(y_test, y_pred_proba[:, 1])`-аар тооцоологдож, **консол дээр хэвлэгдэнэ**. JSON/файлд хадгалагдаагүй; дахин сургахад консолоос авна.
- **F1 (болон precision, recall):** `classification_report(y_test, y_pred_best)`-аар тооцоологдож, **консол дээр “Classification Report”** хэлбэрээр хэвлэгдэнэ. Тайланд оруулахын тулд сургалтыг дахин ажиллуулж энэ гаралтыг хуулж авах эсвэл скриптэд F1/AUC-г JSON-д нэмж хадгалах боломжтой.

**Дүгнэлт:** Accuracy бодитоор ~98%; AUC болон F1 нь сургалтын консол гаралтад байгаа, одоогоор файлд бүртгэгдээгүй.

### Train/test split

- **Одоогийн хэрэгжүүлэлт:** **Time-based split** хийгдсэн.
  - `TEST_YEAR_START = 2023`: **2015–2022** (эсвэл өгөгдөл байгаа өмнөх жилүүд) → **train**; **2023 ба түүнээс хойш** → **test**.
  - Жилүүд хангалтгүй бол (жишээ нь зөвхөн нэг жил) **random split** (80/20) fallback ашиглана.
- **Өмнө** (сайжруулалт хийхээс өмнө): Random split (train_test_split, test_size=0.2, stratify=y) ашигладаг байсан.

### Dataset хэмжээ

| Өгөгдөл | Хэмжээ |
|--------|--------|
| **dzud_ai_dataset_advanced.csv** (сургалтын датасет) | ~**510 мөр** (header-ийг тооцвол 511 мөр) |
| **weather_omnogovi_monthly_clean.csv** (цаг агаар) | ~**3060 мөр** |
| **Feature-ийн тоо** | **24** (avg_temp, min_temp, wind_speed, snowfall_sum, precip_sum, lag1×5, lag2×5, is_winter, cold_index, snow_cumulative, precip_deficit, extreme_cold, extreme_wind, heavy_snow, total_livestock, livestock_change_pct) |
| **Target** | `target_dzud` (0/1 — зудын жил эсэх) |

---

## III. Архитектурын мэдээлэл

### Backend

- **Flask** ашигласан (Python).
- **MVC бүтэц:** Хатуу MVC биш; ойролцоогоор:
  - **Controller/Route:** `app.py` — route-ууд (`/`, `/api/predict`, `/api/risk-map`, `/api/locations`, `/api/geojson`, `/api/health`).
  - **Model/Logic:** `risk_predictor.py` — `DzudRiskPredictor` класс (эрсдэл тооцоолол, ML feature бэлтгэл, hybrid нэгтгэл).
  - **View:** `templates/index.html` — нэг HTML template (JSON хариуг JS-ээр DOM руу гаргана).

### Frontend

- **Pure HTML + CSS + JavaScript** (нэг хуудас).
- **Bootstrap ашиглаагүй** — өөрөө бичсэн CSS (grid, gradient, form, map container).
- **Газрын зураг:** **Leaflet** (v1.9.4) ашигласан:
  - OpenStreetMap tile (`https://{s}.tile.openstreetmap.org/...`).
  - Нэг цэгийн байршил (marker), risk map горимд сумдын polygon (GeoJSON)-ийг өнгөөр харуулна.

### API

- **RESTful** загвар (GET/POST).
- **JSON:** Request body болон response бүгд JSON.
- Endpoint-ууд:
  - `GET /` — вэб интерфэйс (HTML).
  - `POST /api/predict` — нэг байршил + малын тоо + сар → эрсдэл, шалтгаан, зөвлөмж.
  - `POST /api/risk-map` — бүх сумын эрсдэл (газрын зурагт зориулсан).
  - `GET /api/locations` — сумдын жагсаалт.
  - `GET /api/geojson` — сумын хил (GeoJSON).
  - `GET /api/health` — статус, model ачаалагдсан эсэх, weather мөрийн тоо.

### Deployment

- **Одоо:** **Localhost** дээр ажилладаг (Flask development server, `host='0.0.0.0', port=5001`).
- **Production server** (VPS, cloud) дээр байршуулаагүй; албан ёсны deployment diagram байхгүй.

### Өгөгдлийн хадгалалт

- **Database байхгүй** — бүх өгөгдөл **локал файл**:
  - CSV: weather, livestock, dzud_ai_dataset_advanced.
  - GeoJSON: сумын хил.
  - Model: .pkl, .json (metadata).

---

## IV. Use Case — системийн гол хэрэглэгчид

Дараах хэрэглэгчид **багтах**:

| Хэрэглэгчийн төрөл | Багтах эсэх | Тайлбар |
|--------------------|-------------|---------|
| **Малчин** | ✅ Багтана | Байршил + малын тоо оруулан нэг цэгийн эрсдэл, зөвлөмж авах |
| **Аймаг/сумын онцгой комисс** | ✅ Багтана | Risk map (бүх сумын эрсдэл) харах, төлөвлөлт хийх |
| **Судлаач** | ✅ Багтана | REST API ашиглан өгөгдөл татах, судалгаанд ашиглах |
| **Админ** | ❌ Тусгай админ байхгүй | Нэвтрэлт, хэрэглэгчийн удирдлага, контентын удирдлага гэх мэт функц байхгүй |

---

## V. Технологийн судалгаа хэсэгт оруулах зүйлс

Дараах харьцуулалтуудыг тайланд **харьцуулж судлах** боломжтой:

| Сэдэв | Агуулга (санал) |
|-------|------------------|
| **Rule-based vs ML** | Rule-based-ийн тайлбарлах чадвар, ML-ийн нарийвчлал; hybrid (50/50) нэгтгэлийн үндэслэл |
| **Logistic vs Random Forest vs Gradient Boosting** | Гурвыг accuracy (мөн AUC, F1) харьцуулах; одоогоор GB сонгогдсон шалтгаан |
| **Flask vs Django** | Хөнгөн API-д Flask сонгосон шалтгаан; Django-ийн давуу тал (admin, ORM) |
| **ERA5 vs Уламжлалт цаг агаарын станцын өгөгдөл** | ERA5 reanalysis-ийн давуу сул тал, нөөц, нарийвчлал |
| **SHAP / Explainable AI** | Одоо хэрэгжүүлээгүй; тайланд “цаашдын хөгжүүлэлт” болгон SHAP-аар feature importance тайлбарлах санал оруулж болно |

---

## VI. Зураг, диаграмм — тайланд оруулах зүйлс

Дараах диаграммуудыг **оруулах нь тохиромжтой** (одоо кодонд байгаа бүтэц, урсгалд суурилсан):

| Диаграмм | Агуулга (товч) |
|----------|-----------------|
| **System Architecture diagram** | Frontend (HTML/JS/Leaflet) ↔ Flask API ↔ DzudRiskPredictor ↔ CSV/GeoJSON + .pkl |
| **Use Case diagram (UML)** | Actor: Малчин, Комисс, Судлаач; Use case: Эрсдэл шалгах, Risk map харах, API ашиглах |
| **Data flow diagram** | Оролт (lat, lon, livestock, month) → Preprocess → Rule + ML → Эрсдэлийн оноо/түвшин/зөвлөмж → JSON хариу |
| **Model training pipeline** | weather CSV + livestock CSV → make_dataset_advanced → dzud_ai_dataset_advanced.csv → train_model_advanced (time split, 3 model, best .pkl) |
| **Database structure** | Одоо DB байхгүй — “Файл суурьтай өгөгдөл (CSV, GeoJSON)” гэсэн диаграмм (entity: Weather, Livestock, Soum, Model) |
| **Deployment diagram** | Одоо: Хэрэглэгч → Browser → localhost:5001 (Flask) → локал файлууд; Цаашид: server/cloud, Gunicorn гэх мэт |
| **Risk scoring logic flowchart** | Координат → Ойр сум олох → Цаг агаар авах → Weather risk (rule) + Livestock exposure → Rule score; (ML байвал) Feature бэлтгэх → ML prob → Hybrid = 0.5×rule + 0.5×ml → Түвшин/өнгө |

Эдгээрийг draw.io, Lucidchart, PowerPoint эсвэл Mermaid гэх мэтээр зурж тайланд оруулна.

---

## Товч хүснэгт (шууд хуулж ашиглах)

| Асуулт | Хариулт |
|--------|---------|
| Production дээр зөвхөн rule-based уу? | Үгүй; ML файл байвал **hybrid** (rule + ML 50/50) ажиллана. |
| ML модель ашигладаг уу? | Тийм, .pkl байгаа бол predict()-д ашиглагдана. |
| Ямар алгоритм туршсан бэ? | Logistic Regression, Random Forest, Gradient Boosting. |
| Accuracy бодитоор хэд вэ? | ~98.04% (Gradient Boosting, metadata). |
| AUC, F1? | Сургалтын консол дээр тооцоологдож хэвлэгдэнэ; файлд хадгалагдаагүй. |
| Train/test split? | **Time-based**: train &lt; 2023, test ≥ 2023. Fallback: random 80/20. |
| Dataset мөр, feature? | ~510 мөр, **24** feature. |
| Backend? | **Flask**; MVC-т ойр бүтэц. |
| Frontend? | **Pure HTML/CSS/JS**, **Leaflet** (map), Bootstrap ашиглаагүй. |
| API? | **RESTful**, **JSON**. |
| Deployment? | **Localhost** (Flask dev server, port 5001); server дээр байршуулаагүй. |
| Хэрэглэгчид? | Малчин, Аймаг/сумын комисс, Судлаач багтана; Админ тусгай байхгүй. |
