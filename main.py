import pandas as pd
from src.ingest import DataIngestion
from src.profile_engine import ProfileEngine
from src.officer_profile import OfficerProfile
from src.policy_rules import PolicyEngine
from src.explainer import AlertExplainer

def run_poc_simulation():
    print("Initializing Employee Digital Twin POC...")

    # 1. Ingest historical data and train the AI model
    ingest = DataIngestion(data_dir="data")
    historical_df, historical_features = ingest.load_historical_data()
    
    engine = ProfileEngine()
    engine.train(historical_features)

    # 2. Load live test sessions (current_sessions.csv)
    _, current_features_df = ingest.load_current_data()
    
    # Let's inspect a few specific live sessions (e.g., Normal session vs Malicious session)
    print(f"\nSuccessfully loaded {len(current_features_df)} live test sessions for evaluation.\n")

    policy_engine = PolicyEngine()
    explainer = AlertExplainer()

    # 3. Simulate processing each live session through our multi-layer twin
    for index, row in current_features_df.iterrows():
        officer_id = f"OFF_00{index + 1}" if index < 5 else f"OFF_005"
        role = "Senior Investigator" if index in [3, 4] else "Junior Analyst"
        
        session_data = row.to_dict()

        # A. Initialize personalized officer profile baseline
        profile = OfficerProfile(officer_id=officer_id, role=role)
        profile.build_baseline(historical_df)

        # B. Run Personal Profile Check
        profile_result = profile.evaluate_session_context(session_data)

        # C. Run Global AI Anomaly Engine (Isolation Forest)
        ml_result = engine.score_session(session_data)

        # D. Run Deterministic Policy Checks
        policy_result = policy_engine.evaluate_policies(role, session_data)

        # E. Generate Natural Language Explanation Report if risk or anomaly is found
        if ml_result["is_global_anomaly"] or profile_result["personal_risk_score"] > 0 or policy_result["policy_violations"]:
            print(f"=== EVALUATING SESSION {session_data.get('session_id', index)} ({officer_id}) ===")
            report = explainer.generate_summary(officer_id, role, ml_result, profile_result, policy_result)
            print(report)
            print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    run_poc_simulation()