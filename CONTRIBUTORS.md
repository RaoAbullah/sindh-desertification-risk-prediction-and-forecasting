# Contributors & Project Governance

This project is the culmination of an equal collaborative partnership (50/50 contribution) developed as a Final Year Project (FYP) in Computer Science & Engineering.

---

## 👥 Core Authors & Project Leads

| Contributor | Role | GitHub | Contact / Profile |
| :--- | :--- | :---: | :---: |
| **Rao Abdullah** | Co-Lead Researcher & Core Systems Architect | [@RaoAbullah](https://github.com/RaoAbullah) | [github.com/RaoAbullah](https://github.com/RaoAbullah) |
| **Ahmed Faraz Shaikh** | Co-Lead Researcher & Machine Learning Engineer | [@ahmedfarazsyk](https://github.com/ahmedfarazsyk) | [github.com/ahmedfarazsyk](https://github.com/ahmedfarazsyk) |

---

## 🛠️ Contribution Breakdown & Technical Modules

Both co-authors contributed equally across the complete research lifecycle, architectural engineering, deep learning model development, web platform deployment, and academic thesis writing:

### 1. Geospatial Data Engineering & Remote Sensing Pipeline
- **Google Earth Engine (GEE API) Ingestion:** Scripted automated spatial extraction pipelines for Sentinel-1 C-band SAR, Sentinel-2 Multispectral surface reflectance, MODIS Land Surface Temperature (LST) and Evapotranspiration, CHIRPS rainfall, TerraClimate indicators, SoilGrids pedological maps, and SRTM elevation.
- **Stratified Disproportionate Sampling Framework:** Designed the spatial sampling grid across Sindh's 4 major agro-ecological zones (1,000 patches of $640\text{ m} \times 640\text{ m}$ at $10\text{ m}$ ground sampling distance).
- **38-Channel Feature Engineering:** Formulated the continuous physical indices (NDVI, NDWI, MSI, NDDI, SAVI, EVI) and standardized the Mediterranean Desertification and Land Use (**MEDALUS**) quality indices (CQI, VQI, SQI, LQI) to generate target Desertification Sensitivity Index (DSI) ground truth maps.
- **Quality Control & Imputation:** Implemented cloud/shadow QA masking, speckle filtering, and temporal tensor concatenation over 122 observation months ($2016 \to 2026$).

### 2. Deep Learning Research & Model Architectures
- **Model A (Spatio-Temporal UNet Estimator):**
  - Architected symmetric 4-stage convolutional UNet with Group Normalization ($G = \max(1, C/8)$) and smooth SiLU activations to process 33 multi-sensor input channels.
  - Formulated the exponentially tiered DSI-weighted compound loss function ($\alpha\text{-MSE} + (1-\alpha)\text{MAE}$) prioritizing critical degradation zones ($DSI \ge 0.6$).
  - Conducted stratified residual analysis and input perturbation sensitivity tests to quantify environmental driver attributions.
- **Model B (ConvGRU-UNet Seq2Seq Forecaster):**
  - Developed spatial-temporal sequence-to-sequence neural network integrating convolutional Gated Recurrent Units (ConvGRU) within an encoder-decoder topology to preserve 2D feature map geometry inside recurrence.
  - Implemented scheduled sampling (teacher forcing decay) and sliding quarterly window datasets ($2016 \to 2023$).
  - Performed temporal error drift analysis and predictive uncertainty mapping via Test-Time Augmentation (TTA).

### 3. FastAPI Microservices & Backend Engineering
- **Production REST API:** Designed asynchronous route handlers (`/api/v1/analyze/current`, `/api/v1/analyze/forecast`, `/api/v1/search`, `/api/v1/district/{name}`, `/api/v1/history`).
- **Dual-Database Persistence Layer:** Structured MongoDB collections for high-speed retrieval of precomputed administrative district baselines and Supabase tables for on-demand user bounding box analysis sessions.
- **Automated Diagnostic Plotting:** Built publication-quality visualization routines generating 4-panel diagnostic dashboards, 33-channel spatial grids, bivariate hexbin distributions, and animated multi-year timelapse GIFs.
- **Multimodal Generative AI Copilot & Reporting:** Integrated Google Gemini (1.5/2.0 Flash) to deliver real-time token-streaming conversational RAG and automated executive policy assessment briefs.

### 4. Next.js 16 & React 19 Geospatial Dashboard
- **Interactive Mapping UI:** Built full-screen responsive mapping interface utilizing Leaflet, React-Leaflet, and `leaflet-draw` for interactive administrative district selections and arbitrary user-drawn bounding boxes.
- **Time-Series Visualization & Player:** Implemented animated GIF player with frame scrubbing, temporal anomaly maps, and seasonal variance curves.
- **Client-Side Export:** Integrated `html2pdf.js` for instant PDF document rendering and export of generated policy reports.

### 5. Academic Publications & Thesis Documentation
- Prepared the Final Year Project (FYP) thesis, comprehensive framework documentation, methodology diagrams, experimental ablation analyses, and conference paper manuscripts.

---

## 📜 Academic Citation

```bibtex
@misc{abdullah_faraz_2026_stafdrs,
  author       = {Rao Abdullah and Ahmed Faraz Shaikh},
  title        = {STAFDRS: Spatio-Temporal AI Framework for Sindh Desertification Risk Prediction and Forecasting},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/RaoAbullah/sindh-desertification-risk-prediction-and-forecasting}}
}
```
