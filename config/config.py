# A-DIDS Configuration for UAE 2026 Showcase

# Default network features for the ISOT Drone Dataset
ISOT_FEATURE_NAMES = [
    "Payload_Length", "Packet_Rate", "Flow_Duration", "Protocol_Type",
    "Source_IP_Entropy", "Destination_IP_Entropy", "Packet_Size_Avg",
    "TCP_Flags", "Inter_Arrival_Time"
] + [f"Feature_{i}" for i in range(10, 62)]

# Output Paths
MODEL_SAVE_PATH = "A-DIDS/models/tslt_net_v1"
ONNX_EXPORT_PATH = "A-DIDS/models/tslt_net_v1.onnx"
XAI_REPORT_PATH = "A-DIDS/data/xai_summary.png"

# Deployment Unit Info
DEPLOYMENT_UNIT = "UAE-Tactical-Swarm-01"
REGION = "Abu Dhabi Defence Sector"
