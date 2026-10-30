# 🥦 AI-Powered Food Supply Chain Spoilage Prediction System
🔗 **Live Demo:** 
https://al--powered-food-supply-chain-spoilage-prediction-system-uihkw.streamlit.app/

A machine-learning system that predicts **spoilage risk** and **remaining shelf life** of perishable food in the cold chain, using sensor-style data (temperature, humidity, transit time, CO₂, ethylene, door-open events, vibration, packaging).

## Features
- Synthetic IoT / cold-chain dataset generator (9 products, 4 packaging types)
- **Spoilage classifier** (Gradient Boosting) → probability + LOW / MEDIUM / HIGH risk
- **Shelf-life regressor** (Random Forest) → remaining days
- Actionable recommendations (FEFO dispatch, divert, discard)
- **Streamlit dashboard** (single shipment + batch CSV)
- **FastAPI REST API** (`/predict`)
- CLI inference and unit tests

## Project Structure
```
├── app.py                 # Streamlit dashboard
├── api.py                 # FastAPI service
├── src/
│   ├── config.py          # constants, features, product profiles
│   ├── generate_data.py   # synthetic dataset generator
│   ├── train.py           # training + evaluation + feature importance plot
│   └── predict.py         # inference helpers + CLI
├── tests/test_pipeline.py
├── data/  models/         # generated artifacts
└── requirements.txt
```

## Quick Start
```bash
git clone https://github.com/shivommishra27/Al--Powered-food-Supply-Chain-Spoilage-Prediction-System.git
cd Al--Powered-food-Supply-Chain-Spoilage-Prediction-System
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

python -m src.generate_data     # create dataset
python -m src.train             # train models + metrics
python -m src.predict           # sample prediction

streamlit run app.py            # dashboard
uvicorn api:app --reload        # API docs at http://127.0.0.1:8000/docs
pytest                          # run tests
```

## Example API Call
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "product":"Milk","packaging":"Standard","avg_temp_c":9,"transit_hours":36,"days_since_harvest":2
}'
```

## Results (synthetic test set)
| Model | Metric | Score |
|---|---|---|
| Classifier | Accuracy / F1 / ROC-AUC | ~0.90 / ~0.86 / ~0.96 |
| Regressor | MAE / R² | ~1.0 day / ~0.97 |

> Data is simulated using an exponential temperature-decay model. To use real data, replace `data/supply_chain_data.csv` with a CSV having the same columns and rerun `python -m src.train`.

## Future Work
- Real IoT sensor streaming (MQTT) and LSTM time-series models
- Route optimisation and dynamic pricing for near-expiry stock
- Docker deployment and cloud dashboard

## License
MIT
