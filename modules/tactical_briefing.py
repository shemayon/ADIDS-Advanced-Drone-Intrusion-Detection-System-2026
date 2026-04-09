import json

class TacticalBriefing:
    def __init__(self, language="en"):
        """
        Tactical Briefing Module for A-DIDS.
        Simulates an LLM agent that interprets IDS/XAI data into narrative reports.
        
        Args:
            language (str): 'en' or 'ar' (Arabic)
        """
        self.language = language
        self.templates = {
            "en": {
                "header": "### [A-DIDS] TACTICAL SECURITY ALERT",
                "alert": "Alert Type: {alert_type}",
                "confidence": "Confidence Score: {confidence:.2%}",
                "narrative": "Analysis: The network flow for {unit_id} exhibits abnormal behavior characteristic of an {alert_type} attack. This assessment is driven by anomalies in {features}.",
                "recommendation": "Recommended Action: {recommendation}",
                "footer": "--- Deployment Unit: UAE-Tactical-Edge ---"
            },
            "ar": {
                "header": "### [A-DIDS] تنبيه أمني تكتيكي",
                "alert": "نوع التنبيه: {alert_type}",
                "confidence": "درجة الثقة: {confidence:.2%}",
                "narrative": "التحليل: أظهر تدفق الشبكة للوحدة {unit_id} سلوكاً غير طبيعي يشير إلى هجوم من نوع {alert_type}. هذا التقييم مدفوع بشذوذ في {features}.",
                "recommendation": "الإجراء الموصى به: {recommendation}",
                "footer": "--- وحدة الانتشار: الإمارات - الحافة التكتيكية ---"
            }
        }
        
        self.attack_decoding = {
            "DoS": {"ar": "هجوم حجب الخدمة", "rec_en": "Trigger frequency hopping.", "rec_ar": "بدء القفز الترددي الطارئ."},
            "Spoofing": {"ar": "هجوم انتحال الهوية", "rec_en": "Initiate cryptographic re-handshake.", "rec_ar": "بدء إعادة المصافحة المشفرة."},
            "Injection": {"ar": "حقن البيانات", "rec_en": "Isolate the compromised node.", "rec_ar": "عزل العقدة المخترقة."},
            "Normal": {"ar": "طبيعي", "rec_en": "Continue baseline monitoring.", "rec_ar": "متابعة المراقبة الأساسية."}
        }

    def generate_briefing(self, alert_type, confidence, top_features, unit_id="Drone-04"):
        """
        Generates a tactical narrative report.
        """
        # Translate alert type for Arabic if needed
        translated_type = alert_type
        if self.language == "ar" and alert_type in self.attack_decoding:
            translated_type = self.attack_decoding[alert_type]["ar"]
            
        # Get recommendations
        rec_key = f"rec_{self.language}"
        recommendation = self.attack_decoding.get(alert_type, {}).get(rec_key, "Alert command.")
        
        # Format features string
        features_str = ", ".join(top_features)
        
        tpl = self.templates[self.language]
        
        report = [
            tpl["header"],
            tpl["alert"].format(alert_type=translated_type),
            tpl["confidence"].format(confidence=confidence),
            tpl["narrative"].format(unit_id=unit_id, alert_type=translated_type, features=features_str),
            tpl["recommendation"].format(recommendation=recommendation),
            tpl["footer"]
        ]
        
        return "\n".join(report)

if __name__ == "__main__":
    brief = TacticalBriefing(language="en")
    print(brief.generate_briefing("DoS", 0.998, ["Packet_Rate", "Payload_Length"]))
    
    print("\n")
    
    brief_ar = TacticalBriefing(language="ar")
    print(brief_ar.generate_briefing("DoS", 0.998, ["معدل الحزم", "طول الحمولة"]))
