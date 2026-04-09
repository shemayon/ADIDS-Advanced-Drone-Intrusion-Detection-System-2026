import pandas as pd
import numpy as np
import os
from scapy.all import wrpcap, IP, TCP, UDP, Raw
import random

def generate_demo_csv(output_path):
    print(f"[INFO] Generating demo CSV at {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    features = [
        'Duration', 'Rate', 'Entropy',
        'Payload_Length', 'Var_Payload',
        'syn_flag_number', 'ack_flag_number',
        'rst_flag_number', 'fin_flag_number',
        'TCP', 'UDP',
        'Number', 'Tot size',
        'AVG', 'Std', 'Min', 'Max'
    ]
    
    # Generate 1000 records
    n_samples = 1000
    data = np.random.randn(n_samples, len(features))
    
    df = pd.DataFrame(data, columns=features)
    
    # Add attack_type and label
    df['attack_type'] = ['benign'] * 800 + ['DoS'] * 200
    df['label'] = (df['attack_type'] != 'benign').astype(int)
    
    # Inject some correlation for the model to learn
    # DoS usually has high Rate and Number of packets
    df.loc[df['label'] == 1, 'Rate'] += 5.0
    df.loc[df['label'] == 1, 'Number'] += 10.0
    df.loc[df['label'] == 1, 'Duration'] += 2.0
    
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Demo CSV created with {n_samples} rows.")

def generate_demo_pcap(output_path):
    print(f"[INFO] Generating demo PCAP at {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    pkts = []
    # Generate 50 packets representing 2 different flows
    for _ in range(25):
        # Flow 1: Benign DNS request
        p = IP(src="192.168.10.5", dst="192.168.10.1") / UDP(sport=53, dport=random.randint(40000, 60000)) / Raw(load="DNS Query Content")
        pkts.append(p)
        
    for _ in range(25):
        # Flow 2: Potential DoS (High frequency TCP SYN)
        p = IP(src="10.0.0.50", dst="192.168.10.1") / TCP(sport=random.randint(1024, 65535), dport=80, flags="S") / Raw(load="A"*10)
        pkts.append(p)
        
    wrpcap(output_path, pkts)
    print(f"[SUCCESS] Demo PCAP created with {len(pkts)} packets.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    csv_dir = os.path.join(project_root, "A-DIDS/data/extracted/ISOT Drone Dataset/CSV_Format/new_feature_csv/60/benign")
    generate_demo_csv(os.path.join(csv_dir, "demo_features.csv"))
    
    pcap_dir = os.path.join(project_root, "A-DIDS/data/extracted/ISOT Drone Dataset/Raw Data/DoS")
    generate_demo_pcap(os.path.join(pcap_dir, "Dos_1_4_30_20mins.pcap"))
