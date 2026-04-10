"""
modules/temporal_memory.py  —  Phase 8: Time-Series Memory
Moves beyond single-flow detection by preserving a sliding window of recent
network flows. Finds insidious persistent threats (APT) that hide in slow drips.
"""
from typing import Dict, List
import numpy as np

class TemporalMemory:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = []
        print(f"[Temporal] Sliding Window Sequence Analyzer Initialized (n={window_size}).")
        
    def add_flow(self, flow: Dict[str, float]) -> float:
        """
        Adds flow to memory. Returns an 'escalation score' based on flow trajectory.
        """
        self.history.append(flow)
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        # Example logic: If the packet rate is slowly creeping up across the window,
        # it indicates a Low-and-Slow attack attempting to bypass static thresholds.
        rates = [f.get("Rate", 0) for f in self.history]
        if len(rates) == self.window_size:
            trend = np.polyfit(range(self.window_size), rates, 1)[0]
            if trend > 0.5:
                print(f"  [Temporal] ⚠️ Sequence Escalation: Packet rate steadily climbing (+{trend:.2f}/flow).")
                return 1.0 # High escalation
        return 0.0 # Normal
