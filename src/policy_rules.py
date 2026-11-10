class PolicyEngine:
    def __init__(self):
        # Hard limits defined by bank compliance policy
        self.MAX_ALLOWED_EXPORT_KB = 5000.0

    def evaluate_policies(self, officer_role: str, session_data: dict) -> dict:
        """
        Runs deterministic rule checks against the session.
        Returns policy violation flags and whether a hard block is required.
        """
        violations = []
        is_hard_blocked = False

        # Rule 1: Junior analysts attempting overrides
        if officer_role == "Junior Analyst" and session_data.get("override_attempts", 0) > 0:
            violations.append("POLICY VIOLATION: Junior Analyst attempted unauthorized portfolio override.")
            is_hard_blocked = True

        # Rule 2: Hard ceiling on data exports
        export_vol = session_data.get("export_volume_kb", 0.0)
        if export_vol > self.MAX_ALLOWED_EXPORT_KB:
            violations.append(f"POLICY VIOLATION: Data export ({export_vol} KB) exceeded the absolute regulatory limit ({self.MAX_ALLOWED_EXPORT_KB} KB).")
            is_hard_blocked = True

        # Rule 3: Skipped evidence checks on sensitive cases
        skipped_checks = session_data.get("skipped_evidence_checks", 0)
        if skipped_checks > 3:
            violations.append(f"POLICY VIOLATION: Excessive skipped mandatory evidence checks ({skipped_checks} counts).")

        return {
            "policy_violations": violations,
            "is_hard_blocked": is_hard_blocked
        }