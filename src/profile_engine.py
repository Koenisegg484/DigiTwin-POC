import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

class ProfileEngine:
    def __init__(self, model_path="models/isolation_forest.pkl"):
        self.model_path = model_path
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.feature_columns = [
            "login_hour",
            "session_duration_min",
            "is_unusual_ip",
            "cases_reviewed_count",
            "export_volume_kb",
            "skipped_evidence_checks",
            "override_attempts"
        ]

    def train(self, historical_features_df: pd.DataFrame):
        """
        Trains the Isolation Forest model on historical baseline data.
        """
        X = historical_features_df[self.feature_columns]
        self.model.fit(X)
        self.save_model()
        print("[ProfileEngine] Isolation Forest model successfully trained and saved.")

    def save_model(self):
        """Saves the trained model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def load_model(self):
        """Loads a pre-trained model from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False

    def score_session(self, session_features: dict) -> dict:
        """
        Evaluates a single session feature dictionary using the Isolation Forest.
        Returns anomaly score and risk classification.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Model file not found. Train the model first.")
        
        self.load_model()

        # Convert dictionary to dataframe row matching feature columns
        df_input = pd.DataFrame([session_features])[self.feature_columns]

        # Isolation Forest prediction: 1 = normal, -1 = anomaly
        prediction = self.model.predict(df_input)[0]
        
        # decision_function returns anomaly score (lower means more anomalous)
        raw_score = self.model.decision_function(df_input)[0]
        
        # Normalize score into a 0.0 to 1.0 risk probability metric
        # (Lower decision function values map to higher risk)
        # normalized_risk = float(1.0 / (1.0 + pd.np.exp(raw_score * 3)))
        normalized_risk = float(1.0 / (1.0 + np.exp(raw_score * 3)))

        is_anomaly = True if prediction == -1 else False

        return {
            "is_global_anomaly": is_anomaly,
            "anomaly_score": round(normalized_risk, 4),
            "raw_decision_value": round(float(raw_score), 4)
        }

# if __name__ == "__main__":
#     # Quick standalone test
#     from src.ingest import DataIngestion
    
#     ingest = DataIngestion()
#     _, historical_features = ingest.load_historical_data()
    
#     engine = ProfileEngine()
#     engine.train(historical_features)
    
#     # Test a malicious session vector
#     test_vector = {
#         "login_hour": 2,
#         "session_duration_min": 15,
#         "is_unusual_ip": 1,
#         "cases_reviewed_count": 0,
#         "export_volume_kb": 8500.0,
#         "skipped_evidence_checks": 2,
#         "override_attempts": 1
#     }
    
#     result = engine.score_session(test_vector)
#     print("Isolation Forest Evaluation Result:", result)