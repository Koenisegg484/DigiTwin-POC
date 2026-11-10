import numpy as np
import pandas as pd

class OfficerProfile:
    def __init__(self, officer_id: str, role: str):
        self.officer_id = officer_id
        self.role = role
        self.baseline_stats = {}
        self.is_initialized = False

    def build_baseline(self, historical_df: pd.DataFrame):
        """
        Learns the officer's personal baseline behavior from historical data.
        Calculates mean and thresholds for key metrics.
        """
        officer_history = historical_df[historical_df["officer_id"] == self.officer_id]
        
        if officer_history.empty:
            # Fallback to role-based default if no direct history exists
            self.baseline_stats = {
                "avg_login_hour": 9.0,
                "max_export_kb": 150.0,
                "avg_cases_reviewed": 20.0,
                "allow_overrides": False
            }
        else:
            self.baseline_stats = {
                "avg_login_hour": float(officer_history["login_hour"].mean()),
                "max_export_kb": float(officer_history["export_volume_kb"].max()),
                "avg_cases_reviewed": float(officer_history["cases_reviewed_count"].mean()),
                "allow_overrides": True if self.role in ["Senior Investigator", "Compliance Manager"] else False
            }
        
        self.is_initialized = True

    def evaluate_session_context(self, session_data: dict) -> dict:
        """
        Compares an incoming live session against this specific officer's baseline.
        Returns a dictionary of personalized deviation flags.
        """
        if not self.is_initialized:
            raise ValueError(f"Profile for {self.officer_id} has not been initialized with a baseline.")

        deviations = []
        risk_increment = 0.0

        # Check 1: Login Hour Deviation
        current_hour = session_data.get("login_hour", 9)
        expected_hour = self.baseline_stats["avg_login_hour"]
        if abs(current_hour - expected_hour) > 5:  # e.g., logging in 5+ hours away from norm
            deviations.append(f"Unusual login time ({current_hour}:00 vs usual ~{int(expected_hour)}:00)")
            risk_increment += 0.3

        # Check 2: Data Export Spike
        current_export = session_data.get("export_volume_kb", 0.0)
        max_allowed_export = max(self.baseline_stats["max_export_kb"] * 2.0, 500.0) # 2x historical max or 500KB buffer
        if current_export > max_allowed_export:
            deviations.append(f"Abnormal export volume ({current_export} KB exceeds personal threshold)")
            risk_increment += 0.4

        # Check 3: Unauthorized Override Check
        override_attempts = session_data.get("override_attempts", 0)
        if override_attempts > 0 and not self.baseline_stats["allow_overrides"]:
            deviations.append("Attempted privilege override without role authorization")
            risk_increment += 0.5

        return {
            "officer_id": self.officer_id,
            "role": self.role,
            "personal_risk_score": min(risk_increment, 1.0),
            "deviations": deviations
        }

# if __name__ == "__main__":
#     # Quick test of the class
#     mock_history = pd.DataFrame({
#         "officer_id": ["OFF_001", "OFF_001", "OFF_001"],
#         "login_hour": [9, 10, 9],
#         "export_volume_kb": [50.0, 45.0, 60.0],
#         "cases_reviewed_count": [15, 18, 20]
#     })
    
#     officer = OfficerProfile(officer_id="OFF_001", role="Junior Analyst")
#     officer.build_baseline(mock_history)
    
#     test_session = {
#         "login_hour": 3, # 3 AM
#         "export_volume_kb": 5000.0, # Massive export
#         "override_attempts": 1
#     }
    
#     result = officer.evaluate_session_context(test_session)
#     print("Officer Profile Evaluation Result:", result)