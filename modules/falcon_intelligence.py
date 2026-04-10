"""
modules/falcon_intelligence.py  —  Phase 5 (Live AI Brain)
Integrates UAE TII's Falcon models (or a mocked reasoning API for the showcase)
to replace static templates with generative, context-aware command briefings.
"""

import os
import sys
from typing import Dict, Any, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import DEPLOYMENT_UNIT

class FalconIntelligenceEngine:
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self.pipeline = None
        
        if not use_mock:
            print("[Falcon] Loading Falcon-7B model pipeline... (this requires heavy GPU)")
            try:
                from transformers import pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model="tiiuae/falcon-7b-instruct",
                    device_map="auto"
                )
            except ImportError:
                print("[Falcon] Failed to load transformers. Falling back to Mock.")
                self.use_mock = True

    def generate_tactical_reasoning(
        self,
        alert_type: str,
        confidence: float,
        triggers: List[str],
        flow_features: Dict[str, float]
    ) -> str:
        """
        Generates dynamic tactical reasoning based on the live threat.
        """
        prompt = f"""You are Falcon-H1, the AI commander of {DEPLOYMENT_UNIT}.
A {alert_type} attack was detected with {confidence:.1%} confidence.
The XAI engine flagged these metrics: {', '.join(triggers)}.
Rate is {flow_features.get('Rate', 0)} pkts/s and duration is {flow_features.get('Duration', 0)}s.

Provide a concise 2-sentence tactical reasoning in English, followed by its Arabic translation.
Do not use templates. Reason about the metrics.
"""

        if self.use_mock:
            # Generate a realistic "mocked" LLM output for the showcase
            duration = flow_features.get("Duration", 0)
            rate = flow_features.get("Rate", 0)
            
            en = (f"The detection of {alert_type} is confirmed by extreme packet rates "
                  f"({rate:.1f} pkts/s) sustained over {duration:.1f} seconds. "
                  f"Since {triggers[0]} is the primary trigger, this indicates highly structured malicious payloads.")
                  
            ar = (f"تم تأكيد اكتشاف {alert_type} من خلال معدلات حزم متطرفة ({rate:.1f} حزمة/ثانية) "
                  f"مستمرة لفترة {duration:.1f} ثانية. نظرًا لأن {triggers[0]} هو المؤشر الرئيسي، "
                  f"فإن هذا يشير إلى حمولات خبيثة شديدة التنظيم.")
                  
            return f"{en}\n\n{ar}"
        
        # Real pipeline generation
        out = self.pipeline(prompt, max_length=150, do_sample=True, temperature=0.7)
        return out[0]["generated_text"].replace(prompt, "").strip()

if __name__ == "__main__":
    falcon = FalconIntelligenceEngine(use_mock=True)
    report = falcon.generate_tactical_reasoning(
        "DoS", 0.99, ["Entropy", "Rate"], {"Rate": 4500, "Duration": 2.5}
    )
    print(report)
