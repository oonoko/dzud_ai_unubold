# Зудын эрсдэлийн систем — диаграммууд

Доорх диаграммуудыг [Mermaid](https://mermaid.js.org/) ашиглан зурсан. GitHub, VS Code (Mermaid extension), эсвэл [mermaid.live](https://mermaid.live) дээр нээж харна.

---

## 1. System Architecture (Системийн архитектур)

```mermaid
flowchart TB
    subgraph Client["🖥️ Хэрэглэгч"]
        Browser["Browser\n(HTML / CSS / JS)"]
        Leaflet["Leaflet Map\n(OpenStreetMap)"]
        Browser --> Leaflet
    end

    subgraph Backend["⚙️ Backend"]
        Flask["Flask App\n(app.py)"]
        Predictor["DzudRiskPredictor\n(risk_predictor.py)"]
        Flask --> Predictor
    end

    subgraph Data["📁 Өгөгдөл"]
        WeatherCSV["weather_omnogovi_monthly_clean.csv"]
        LivestockCSV["livestock_omnogovi.csv"]
        GeoJSON["omnogovi_soums_simple.geojson"]
        ModelPKL["dzud_risk_model_advanced.pkl\nscaler_advanced.pkl"]
    end

    Browser <-->|"HTTP / JSON\nREST API"| Flask
    Predictor --> WeatherCSV
    Predictor --> LivestockCSV
    Predictor --> ModelPKL
    Flask --> GeoJSON
```

---

## 2. Use Case Diagram (UML)

```mermaid
flowchart LR
    subgraph Actors["Хэрэглэгчид"]
        A1[Малчин]
        A2[Аймаг/сумын\nонцгой комисс]
        A3[Судлаач]
    end

    subgraph System["Зудын эрсдэлийн систем"]
        U1[Эрсдэл шалгах\nкоординат + мал + сар]
        U2[Risk map харах\nбүх сумын эрсдэл]
        U3[API ашиглах\nREST / JSON]
    end

    A1 --> U1
    A1 --> U2
    A2 --> U1
    A2 --> U2
    A3 --> U3
    A3 --> U1
```

**Use Case-ийн дэлгэрэнгүй:**

```mermaid
flowchart TB
    subgraph UC["Use Cases"]
        direction TB
        UC1["UC-1: Эрсдэл шалгах\nPOST /api/predict"]
        UC2["UC-2: Risk map\nPOST /api/risk-map"]
        UC3["UC-3: Сумын жагсаалт\nGET /api/locations"]
        UC4["UC-4: Газрын зураг\nGET /api/geojson"]
    end

    M[Малчин] --> UC1
    M --> UC2
    K[Комисс] --> UC2
    K --> UC3
    S[Судлаач] --> UC3
    S --> UC4
    S --> UC1
```

---

## 3. Data Flow Diagram (Өгөгдлийн урсгал)

```mermaid
flowchart LR
    subgraph Input["Оролт"]
        I1[lat, lon]
        I2[livestock\nsheep, goat, ...]
        I3[month]
    end

    subgraph Process["Боловсруулалт"]
        P1[Ойр сум олох]
        P2[Цаг агаар авах\nCSV]
        P3[Rule-based\nэрсдэл]
        P4[ML feature\nбэлтгэх]
        P5[ML predict_proba]
        P6[Hybrid нэгтгэх\n50/50]
    end

    subgraph Output["Гаралт"]
        O1[Эрсдэлийн оноо\n0-100]
        O2[Түвшин, өнгө]
        O3[Шалтгаан\nтоп 3]
        O4[Зөвлөмж]
    end

    I1 --> P1
    I2 --> P3
    I3 --> P2
    P1 --> P2
    P2 --> P3
    P2 --> P4
    P4 --> P5
    P5 --> P6
    P3 --> P6
    P6 --> O1
    O1 --> O2
    P3 --> O3
    O2 --> O4
```

---

## 4. Model Training Pipeline (Загвар сургах урсгал)

```mermaid
flowchart TB
    subgraph Sources["Эх сурвалж"]
        W[weather_omnogovi_monthly_clean.csv]
        L[livestock_omnogovi.csv]
    end

    subgraph Build["Dataset бэлтгэл"]
        M1[make_dataset_advanced.py]
        Merge[Merge + lag features\n+ target_dzud]
        D[dzud_ai_dataset_advanced.csv\n~510 мөр, 24 feature]
    end

    subgraph Train["Сургалт"]
        M2[train_model_advanced.py]
        Split[Time-based split\ntrain &lt; 2023\ntest ≥ 2023]
        Models[LogisticRegression\nRandomForest\nGradientBoosting]
        Best[Best by accuracy\n→ .pkl]
    end

    subgraph Output["Үр дүн"]
        PKL[dzud_risk_model_advanced.pkl]
        SC[scaler_advanced.pkl]
        META[model_metadata_advanced.json]
    end

    W --> M1
    L --> M1
    M1 --> Merge
    Merge --> D
    D --> M2
    M2 --> Split
    Split --> Models
    Models --> Best
    Best --> PKL
    Best --> SC
    Best --> META
```

---

## 5. Data / File Structure (Өгөгдлийн бүтэц — DB байхгүй)

```mermaid
erDiagram
    Weather ||--o{ Soum : "байрлана"
    Livestock ||--o{ Year : "жил бүр"
    Dataset ||--o{ Weather : "агуулна"
    Dataset ||--o{ Livestock : "агуулна"
    Model ||--o{ Feature : "24 feature"

    Weather {
        string aimag
        string soum
        float lat
        float lon
        int year
        int month
        float avg_temp
        float min_temp
        float wind_speed
        float snowfall_sum
        float precip_sum
    }

    Livestock {
        int year
        float total_livestock
        float livestock_change_pct
        int dzud_year
    }

    Soum {
        string soum
        string aimag
        float lat
        float lon
        geojson boundary
    }

    Dataset {
        string aimag
        string soum
        int year
        int month
        int target_dzud
    }

    Model {
        pkl model_file
        pkl scaler_file
        json metadata
    }
```

**Файлын бүтэц (мод):**

```mermaid
flowchart TB
    root[dzud-ai/]
    root --> app[app.py]
    root --> risk[risk_predictor.py]
    root --> train[train_model_advanced.py]
    root --> make[make_dataset_advanced.py]
    root --> data[Өгөгдөл]
    root --> model[Загвар]
    root --> templates[templates/]

    data --> w[weather_omnogovi_monthly_clean.csv]
    data --> l[livestock_omnogovi.csv]
    data --> d[dzud_ai_dataset_advanced.csv]
    data --> g[omnogovi_soums_simple.geojson]

    model --> pkl[dzud_risk_model_advanced.pkl]
    model --> scaler[scaler_advanced.pkl]
    model --> meta[model_metadata_advanced.json]

    templates --> html[index.html]
```

---

## 6. Deployment Diagram (Суулгах орчин)

```mermaid
flowchart TB
    subgraph User["Хэрэглэгч"]
        U[Хэрэглэгч]
    end

    subgraph Local["Одоогийн орчин - Localhost"]
        Browser[Browser\nhttp://localhost:5001]
        Flask[Flask Dev Server\nport 5001\n0.0.0.0]
        FS[Локал файлууд\nCSV, GeoJSON, .pkl]
    end

    subgraph Future["Цаашдын боломж"]
        Server[VPS / Cloud]
        Gunicorn[Gunicorn\nWSGI]
        Nginx[Nginx]
    end

    U --> Browser
    Browser -->|HTTP| Flask
    Flask --> FS
    U -.->|"цаашид"| Nginx
    Nginx -.-> Gunicorn
    Gunicorn -.-> Server
```

**Дарааллын диаграмм — нэг predict хүсэлт:**

```mermaid
sequenceDiagram
    participant U as Хэрэглэгч
    participant B as Browser
    participant F as Flask API
    participant P as DzudRiskPredictor
    participant C as CSV/Model

    U->>B: Координат, мал, сар оруулах
    B->>F: POST /api/predict (JSON)
    F->>P: predict(lat, lon, livestock, month)
    P->>C: Ойр сум, цаг агаар унших
    C-->>P: Weather row
    P->>P: Rule-based эрсдэл тооцоолох
    P->>C: ML feature, model
    C-->>P: ML probability
    P->>P: Hybrid = 0.5×rule + 0.5×ML
    P-->>F: Result dict
    F-->>B: JSON (risk, reasons, recommendations)
    B-->>U: Эрсдэл, зураг, зөвлөмж харуулах
```

---

## 7. Risk Scoring Logic Flowchart (Эрсдэл тооцоолох логик)

```mermaid
flowchart TB
    Start([Эхлэх]) --> Input[Оролт: lat, lon, livestock, month]
    Input --> FindSoum[Хамгийн ойр сум олох\nfind_nearest_location]
    FindSoum --> GetWeather[Тухайн сум + сарын\nцаг агаар авах\nget_current_weather]
    GetWeather --> Winter{Өвлийн сар уу?\n11,12,1,2,3}
    Winter -->|Үгүй| ZeroRisk[Weather risk = 0\nЗун - эрсдэл байхгүй]
    Winter -->|Тийм| CalcWeather[Цаг агаарын эрсдэл тооцох\nтемп, салхи, цас, хур тунадас\ncalculate_weather_risk]
    CalcWeather --> CalcExposure[Малын өртөлт тооцох\nжин: ямаа/хонь эмзэг\ncalculate_livestock_exposure]
    ZeroRisk --> CalcExposure
    CalcExposure --> RuleScore[Rule-based эцсийн оноо\n0.7×weather + 0.3×exposure\ncalculate_final_risk]
    RuleScore --> HasModel{ML модель\nбайгаа юу?}
    HasModel -->|Үгүй| UseRule[Эцсийн эрсдэл = Rule score\nrisk_source: rule_based]
    HasModel -->|Тийм| BuildFeatures[24 feature бэлтгэх\n_build_ml_features]
    BuildFeatures --> MLPredict[model.predict_proba\nзудын магадлал]
    MLPredict --> Hybrid[Hybrid = 0.5×rule + 0.5×ML×100\nrisk_source: hybrid]
    Hybrid --> UseRule
    UseRule --> Level[Түвшин, өнгө тодорхох\n_score_to_level\nБага/Дунд/Өндөр/Маш өндөр]
    Level --> Rec[Зөвлөмж үүсгэх\nget_recommendations]
    Rec --> End([JSON хариу буцаах])
```

---

## 8. API Endpoint-ууд (товч)

```mermaid
flowchart LR
    subgraph GET["GET"]
        G1[/]
        G2[/api/locations]
        G3[/api/geojson]
        G4[/api/health]
    end

    subgraph POST["POST"]
        P1[/api/predict]
        P2[/api/risk-map]
    end

    Client[Клиент] --> G1
    Client --> G2
    Client --> G3
    Client --> G4
    Client --> P1
    Client --> P2
```

---

Файлыг **GitHub** эсвэл **mermaid.live** дээр нээхэд диаграммууд автоматаар зураг болж харагдана. Word/Google Docs-д оруулахын тулд [mermaid.live](https://mermaid.live) дээр нэг нэгээр нь оруулаад PNG/SVG татаж аваарай.
