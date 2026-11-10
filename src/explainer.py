class AlertExplainer:
    def generate_summary(self, officer_id, role, ml_result, profile_result, policy_result) -> str:
        """
        Combines mathematical anomaly outputs, personal profile deviations,
        and hard policy violations into an actionable human-readable report.
        """
        risk_score = ml_result["anomaly_score"]
        is_anomaly = ml_result["is_global_anomaly"]
        personal_devs = profile_result.get("deviations", [])
        violations = policy_result.get("policy_violations", [])
        is_blocked = policy_result.get("is_hard_blocked", False)

        # Determine overall threat level
        if is_blocked or risk_score > 0.7 or len(violations) > 0:
            severity = "CRITICAL / HIGH RISK"
        elif is_anomaly or risk_score > 0.4 or len(personal_devs) > 0:
            severity = "MODERATE RISK"
        else:
            severity = "LOW RISK (NORMAL)"

        report = f"""
[SECURITY DIGITAL TWIN ALERT REPORT]
--------------------------------------------------
Officer ID   : {officer_id} ({role})
Threat Level : {severity}
ML Risk Score: {risk_score} (Global Anomaly: {is_anomaly})
--------------------------------------------------
[+] Personal Baseline Deviations:
    - {chr(10) + '    - '.join(personal_devs) if personal_devs else 'None detected.'}

[+] Policy Violations:
    - {chr(10) + '    - '.join(violations) if violations else 'None detected.'}

[+] Recommended Action:
    {'>>> ACTION REQUIRED: Session hard-paused and routed to SOC team.' if is_blocked else '>>> Routine logging. Allow session continuation.'}
--------------------------------------------------
"""
        return report.strip()