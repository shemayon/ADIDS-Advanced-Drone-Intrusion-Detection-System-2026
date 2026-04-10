"""
pcap_processor.py  —  A-DIDS PCAP Feature Extractor
Reads a .pcap file using scapy (streaming), groups packets into
bidirectional flows, and computes the 17 features required by the model.

Usage:
    from pcap_processor import process_pcap
    flows = process_pcap("capture.pcap", max_packets=50000)
"""

from __future__ import annotations
import logging
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
logging.getLogger("scapy.loading").setLevel(logging.ERROR)

from scapy.all import PcapReader, Packet
from scapy.layers.inet import IP, TCP
from scapy.layers.inet import UDP as UDP_layer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import FEATURES, PCAP_MAX_PACKETS, PCAP_MIN_FLOW_PKTS


# ── Helpers ───────────────────────────────────────────────────

def _payload_entropy(payloads: List[bytes]) -> float:
    """Shannon entropy of concatenated layer-4 payloads."""
    combined = b"".join(payloads)
    if not combined:
        return 0.0
    freq: Dict[int, int] = {}
    for byte in combined:
        freq[byte] = freq.get(byte, 0) + 1
    n = len(combined)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _flow_key(pkt: Packet) -> Optional[tuple]:
    """Canonical bidirectional 5-tuple (sorts endpoints so A→B = B→A)."""
    if not pkt.haslayer(IP):
        return None
    proto = pkt[IP].proto
    src, dst = pkt[IP].src, pkt[IP].dst
    sport = dport = 0
    if pkt.haslayer(TCP):
        sport, dport = pkt[TCP].sport, pkt[TCP].dport
    elif pkt.haslayer(UDP_layer):
        sport, dport = pkt[UDP_layer].sport, pkt[UDP_layer].dport
    ep_a, ep_b = (src, sport), (dst, dport)
    if ep_a > ep_b:
        ep_a, ep_b = ep_b, ep_a
    return (*ep_a, *ep_b, proto)


def _flow_features(pkts: list) -> Dict[str, float]:
    """Compute all 17 model features from a list of (timestamp, pkt) tuples."""
    times    = [t for t, _ in pkts]
    pkt_list = [p for _, p in pkts]

    duration = (max(times) - min(times)) if len(times) > 1 else 0.0
    number   = float(len(pkt_list))
    rate     = number / duration if duration > 0 else number

    sizes, payloads = [], []
    syn = ack = rst = fin = is_tcp = is_udp = 0

    for p in pkt_list:
        sizes.append(float(len(p)))
        if p.haslayer(TCP):
            is_tcp = 1
            flags  = p[TCP].flags
            if flags & 0x02: syn += 1
            if flags & 0x10: ack += 1
            if flags & 0x04: rst += 1
            if flags & 0x01: fin += 1
            raw = bytes(p[TCP].payload)
        elif p.haslayer(UDP_layer):
            is_udp = 1
            raw    = bytes(p[UDP_layer].payload)
        else:
            raw = b""
        if raw:
            payloads.append(raw)

    sizes_arr = np.array(sizes, dtype=float)
    pl_arr    = np.array([len(pl) for pl in payloads] or [0.0], dtype=float)

    return {
        "Duration":         duration,
        "Rate":             rate,
        "Entropy":          _payload_entropy(payloads),
        "Payload_Length":   float(pl_arr.mean()),
        "Var_Payload":      float(pl_arr.var()),
        "syn_flag_number":  float(syn),
        "ack_flag_number":  float(ack),
        "rst_flag_number":  float(rst),
        "fin_flag_number":  float(fin),
        "TCP":              float(is_tcp),
        "UDP":              float(is_udp),
        "Number":           number,
        "Tot size":         float(sizes_arr.sum()),
        "AVG":              float(sizes_arr.mean()),
        "Std":              float(sizes_arr.std()),
        "Min":              float(sizes_arr.min()),
        "Max":              float(sizes_arr.max()),
    }


# ── Public API ────────────────────────────────────────────────

def process_pcap(
    pcap_path: str,
    min_packets: int = PCAP_MIN_FLOW_PKTS,
    max_packets: int = PCAP_MAX_PACKETS,
) -> List[Dict[str, Any]]:
    """
    Stream-reads a PCAP and returns a list of flow feature dicts.

    Parameters
    ----------
    pcap_path   : path to the .pcap file
    min_packets : discard flows with fewer packets than this
    max_packets : stop after this many packets (0 = read entire file)
    """
    flows: Dict[tuple, list] = defaultdict(list)
    read = 0

    print(f"[PCAP] Reading: {pcap_path}")
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            key = _flow_key(pkt)
            if key is None:
                continue
            flows[key].append((float(pkt.time), pkt))
            read += 1
            if max_packets and read >= max_packets:
                print(f"[PCAP] Reached packet limit ({max_packets:,})")
                break

    print(f"[PCAP] {len(flows):,} raw flows from {read:,} packets")

    results = []
    for key, pkts in flows.items():
        if len(pkts) < min_packets:
            continue
        feats = _flow_features(pkts)
        feats["_flow_key"]   = key
        feats["_pkt_count"]  = len(pkts)
        results.append(feats)

    print(f"[PCAP] {len(results):,} flows after filtering (min={min_packets} pkts)")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap", help="Path to .pcap file")
    ap.add_argument("--max-packets", type=int, default=PCAP_MAX_PACKETS)
    a = ap.parse_args()
    flows = process_pcap(a.pcap, max_packets=a.max_packets)
    print(f"Extracted {len(flows)} flows.")
