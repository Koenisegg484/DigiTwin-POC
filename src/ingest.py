import pandas as pd
import os

class DataIngestion:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    # def load_historical_data(self):
    #     """Loads historical baseline sessions used to train the ML model."""
    #     path = os.path.join(self.data_dir, "historical_sessions.csv")
    #     df = pd.read_csv(path)
    #     return self._preprocess(df)

    def load_historical_data(self):
        """Loads historical baseline sessions used to train the ML model."""
        path = os.path.join(self.data_dir, "historical_sessions.csv")
        df = pd.read_csv(path)
        _, features = self._preprocess(df)
        return df, features

    def load_current_data(self):
        """Loads live current sessions used to test anomaly detection."""
        path = os.path.join(self.data_dir, "current_sessions.csv")
        df = pd.read_csv(path)
        return self._preprocess(df)

    def _preprocess(self, df):
        """
        Feature engineering and cleaning step.
        In production, this is where you would normalize features,
        handle missing values, or scale metrics if needed.
        """
        # Drop non-feature columns that shouldn't go into the math model
        feature_columns = [
            "login_hour",
            "session_duration_min",
            "is_unusual_ip",
            "cases_reviewed_count",
            "export_volume_kb",
            "skipped_evidence_checks",
            "override_attempts"
        ]
        
        # Keep metadata for logging/reporting later
        metadata = df[["session_id", "officer_id", "role"]]
        features = df[feature_columns]
        
        return metadata, features