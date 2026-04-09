# pcap_processor.py

from scapy.all import rdpcap, TCP, UDP, IP
import numpy as np
from collections import defaultdict
import math

def compute_entropy(data):
    if len(data) == 0:
        return 0
    # Probabilities of each byte
    counts = defaultdict(int)
    for b in data:
        counts[b] += 1
    
    data_len = len(data)
    probs = [count / data_len for count in counts.values()]
    return -sum(p * math.log2(p) for p in probs)

def normalize(d):
    return {k: float(v) for k, v in d.items()}

def process_pcap(file_path):
    print(f"[INFO] Processing PCAP: {file_path}")
    packets = rdpcap(file_path)

    flows = defaultdict(list)

    for pkt in packets:
        if IP in pkt:
            # Group by 3-tuple (source, destination, protocol)
            key = (pkt[IP].src, pkt[IP].dst, pkt[IP].proto)
            flows[key].append(pkt)

    features_list = []

    for _, pkts in flows.items():
        times = [float(pkt.time) for pkt in pkts]
        sizes = [len(pkt) for pkt in pkts]

        duration = max(times) - min(times) if len(times) > 1 else 0
        rate = len(pkts) / duration if duration > 0 else 0

        payload = []
        for pkt in pkts:
            if hasattr(pkt, "load"):
                payload.extend(bytes(pkt.load))

        entropy = compute_entropy(payload)

        syn = ack = rst = fin = 0

        for pkt in pkts:
            if TCP in pkt:
                flags = int(pkt[TCP].flags)
                syn += int(flags & 0x02 != 0)
                ack += int(flags & 0x10 != 0)
                rst += int(flags & 0x04 != 0)
                fin += int(flags & 0x01 != 0)

        feature_vector = {
            "Duration": duration,
            "Rate": rate,
            "Entropy": entropy,
            "Payload_Length": np.mean(sizes),
            "Var_Payload": np.var(sizes),
            "syn_flag_number": syn,
            "ack_flag_number": ack,
            "rst_flag_number": rst,
            "fin_flag_number": fin,
            "TCP": int(any(TCP in p for p in pkts)),
            "UDP": int(any(UDP in p for p in pkts)),
            "Number": len(pkts),
            "Tot size": sum(sizes),
            "AVG": np.mean(sizes),
            "Std": np.std(sizes),
            "Min": np.min(sizes),
            "Max": np.max(sizes),
        }

        features_list.append(normalize(feature_vector))

    print(f"[SUCCESS] Extracted features for {len(features_list)} flows.")
    return features_list

if __name__ == "__main__":
    # Test script with a pcap if available
    pass
