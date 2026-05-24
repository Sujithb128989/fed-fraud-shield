from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
import pandas as pd
import pickle
import os

app = FastAPI(title="Federated Anomaly Detection API")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_engine():
    db_url = os.environ.get("DATABASE_URL", "postgresql://user:password@db:5432/federated_db")
    return create_engine(db_url)

@app.get("/api/metrics")
def get_metrics():
    try:
        engine = get_engine()
        metrics_df = pd.read_sql("SELECT * FROM metrics", engine)
        if metrics_df.empty:
            return {"status": "waiting", "data": []}
            
        current_round = int(metrics_df['round_number'].max())
        active_clients = int(metrics_df[metrics_df['round_number'] == current_round]['client_id'].nunique())
        
        # Latest average metrics
        latest_metrics = metrics_df[metrics_df['round_number'] == current_round]
        avg_roc = float(latest_metrics['roc_auc'].mean())
        avg_f1 = float(latest_metrics['f1_score'].mean())
        avg_precision = float(latest_metrics['precision'].mean())
        
        return {
            "status": "active",
            "current_round": current_round,
            "active_clients": active_clients,
            "averages": {
                "roc_auc": avg_roc,
                "f1_score": avg_f1,
                "precision": avg_precision
            },
            "history": metrics_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/shap/{client_id}")
def get_shap(client_id: str):
    engine = get_engine()
    metrics_df = pd.read_sql("SELECT * FROM metrics", engine)
    if metrics_df.empty:
        raise HTTPException(status_code=404, detail="No rounds completed yet")
        
    current_round = int(metrics_df['round_number'].max())
    shap_file = f"/app/data/explanations/{client_id}_round_{current_round}_shap.pkl"
    
    if not os.path.exists(shap_file):
        raise HTTPException(status_code=404, detail="SHAP explanations not found for this client/round")
        
    with open(shap_file, "rb") as f:
        shap_data = pickle.load(f)
        
    shap_values = shap_data['shap_values']
    
    # Handle both list and array formats
    if isinstance(shap_values, list):
        sv = shap_values[0][0]
    else:
        sv = shap_values[0]
        
    features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    # The clients drop Time and Class, so there are 29 features: V1..V28, Amount
    features = [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    if len(sv) != len(features):
        # Fallback if mismatch
        features = [f"Feature {i}" for i in range(len(sv))]
        
    shap_df = pd.DataFrame({
        'feature': features,
        'value': sv
    }).sort_values('value', ascending=False).head(10)
    
    return shap_df.to_dict(orient="records")
