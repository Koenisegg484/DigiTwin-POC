import re
import pandas as pd
from datetime import datetime

# Sample raw logs simulating a compliance system stream
RAW_BANKING_LOGS = [
    "2026-11-02 09:15:10.123 [INFO] [AuthService] User OFF_001 successfully logged in. Client IP: 10.0.1.45 (Known Internal IP)",
    "2026-11-02 09:16:00.450 [INFO] [CaseManagement] Officer OFF_001 opened case ID: CS_8812.",
    "2026-11-02 09:25:30.890 [INFO] [CaseManagement] Officer OFF_001 completed case CS_8812. Evidence checks verified: True.",
    "2026-11-02 14:00:05.111 [INFO] [AuthService] User OFF_002 successfully logged in. Client IP: 10.0.1.52 (Known Internal IP)",
    "2026-11-02 14:05:12.333 [INFO] [CaseManagement] Officer OFF_002 opened and processed 25 routine cases.",
    # Anomalous Session
    "2026-11-02 02:30:15.789 [WARN] [AuthService] User OFF_004 successfully logged in. Client IP: 192.168.4.99 (Unusual/External IP detected)",
    "2026-11-02 02:31:00.010 [INFO] [CaseManagement] Officer OFF_004 opened case ID: CS_9941. Evidence check skipped: True.",
    "2026-11-02 02:31:45.220 [INFO] [CaseManagement] Officer OFF_004 opened case ID: CS_9942. Evidence check skipped: True.",
    "2026-11-02 02:32:45.330 [ERROR] [ExportService] Officer OFF_004 triggered bulk export. Volume: 8500.0 KB.",
    "2026-11-02 02:35:12.900 [WARN] [PolicyEngine] Officer OFF_004 executed override attempt for restricted portfolio."
]

class LogParserPipeline:
    def __init__(self, raw_logs=RAW_BANKING_LOGS):
        self.raw_logs = raw_logs

    def parse_logs_to_dataframe(self):
        """
        Parses raw text logs, groups events by officer session,
        and extracts the exact numerical feature vector required for the model.
        """
        parsed_events = []

        # Regex patterns to extract metadata from log lines
        timestamp_regex = r"^([\d :-.]+)"
        user_regex = r"User\s+(\w+)|Officer\s+(\w+)"
        
        for line in self.raw_logs:
            # Extract timestamp
            ts_match = re.search(timestamp_regex, line)
            timestamp = datetime.strptime(ts_match.group(1).strip(), "%Y-%m-%d %H:%M:%S.%f") if ts_match else datetime.now()
            
            # Extract Officer ID
            user_match = re.search(user_regex, line)
            officer_id = user_match.group(1) or user_match.group(2) if user_match else "UNKNOWN"

            # Parse specific behavioral indicators from the message body
            is_unusual_ip = 1 if "Unusual/External IP" in line else 0
            skipped_evidence = 1 if "Evidence check skipped: True" in line else 0
            
            export_vol = 0.0
            if "Volume:" in line:
                vol_match = re.search(r"Volume:\s*([\d.]+)", line)
                if vol_match:
                    export_vol = float(vol_match.group(1))

            override_attempt = 1 if "override attempt" in line else 0
            case_reviewed = 1 if "opened case" in line or "processed" in line else 0

            parsed_events.append({
                "timestamp": timestamp,
                "officer_id": officer_id,
                "login_hour": timestamp.hour,
                "is_unusual_ip": is_unusual_ip,
                "skipped_evidence_checks": skipped_evidence,
                "export_volume_kb": export_vol,
                "override_attempts": override_attempt,
                "cases_reviewed_count": case_reviewed
            })

        df_raw = pd.DataFrame(parsed_events)

        # Aggregate events by officer into a unified session summary
        session_aggregated = df_raw.groupby("officer_id").agg(
            login_hour=("login_hour", "min"),
            is_unusual_ip=("is_unusual_ip", "max"),
            cases_reviewed_count=("cases_reviewed_count", "sum"),
            export_volume_kb=("export_volume_kb", "sum"),
            skipped_evidence_checks=("skipped_evidence_checks", "sum"),
            override_attempts=("override_attempts", "sum"),
            session_duration_min=("timestamp", lambda x: int((x.max() - x.min()).total_seconds() / 60) + 5) # mock duration
        ).reset_index()

        # Add session IDs for tracking
        session_aggregated.insert(0, "session_id", [f"SES_{i+5000}" for i in range(len(session_aggregated))])
        session_aggregated["role"] = session_aggregated["officer_id"].map(
            lambda x: "Senior Investigator" if x in ["OFF_004", "OFF_005"] else "Junior Analyst"
        )

        return session_aggregated

if __name__ == "__main__":
    parser = LogParserPipeline()
    df_sessions = parser.parse_logs_to_dataframe()
    print("Successfully parsed raw logs into Model-Ready DataFrame:")
    print(df_sessions.to_string())