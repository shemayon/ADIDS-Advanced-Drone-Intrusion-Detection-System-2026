"""
modules/tactical_briefing.py  —  A-DIDS Tactical Reporting Engine
Generates English / Arabic command briefings from IDS detection results
and XAI Truth Triggers. Designed for UAE Military Operator interface.
"""

from __future__ import annotations
import os
import sys
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import DEPLOYMENT_UNIT, REGION


class TacticalBriefing:
    """
    Bilingual Tactical Briefing Engine.
    Converts threat detections + SHAP triggers into narrative reports
    for immediate military decision-making.
    """

    ATTACK_META = {
        "DoS": {
            "en_name": "Denial-of-Service (DoS)",
            "ar_name": "هجوم حجب الخدمة",
            "rec_en":  "Trigger emergency frequency hopping. Isolate affected swarm node.",
            "rec_ar":  "تفعيل القفز الترددي الطارئ. عزل العقدة المتضررة في السرب.",
        },
        "Injection": {
            "en_name": "Command Injection",
            "ar_name": "حقن الأوامر",
            "rec_en":  "Isolate compromised node immediately. Switch to secondary control channel.",
            "rec_ar":  "عزل العقدة الفورية المخترقة. التبديل إلى قناة التحكم الاحتياطية.",
        },
        "Manipulation": {
            "en_name": "Telemetry Manipulation",
            "ar_name": "التلاعب بالقياس عن بُعد",
            "rec_en":  "Cross-validate sensor data with adjacent swarm nodes. Flag for forensics.",
            "rec_ar":  "التحقق المتقاطع من بيانات الاستشعار. وضع علامة للتحليل الجنائي.",
        },
        "MITM": {
            "en_name": "Man-in-the-Middle (MITM)",
            "ar_name": "هجوم الوسيط",
            "rec_en":  "Initiate cryptographic re-handshake. Verify certificate chain.",
            "rec_ar":  "بدء إعادة المصافحة المشفرة. التحقق من سلسلة الشهادات.",
        },
        "Ip Spoofing": {
            "en_name": "IP Spoofing",
            "ar_name": "انتحال عنوان IP",
            "rec_en":  "Activate strict source verification. Blacklist spoofed IP range.",
            "rec_ar":  "تفعيل التحقق الصارم من المصدر. إدراج نطاق IP المنتحل في القائمة السوداء.",
        },
        "Password Cracking": {
            "en_name": "Password Cracking / Brute-Force",
            "ar_name": "كسر كلمة المرور / القوة الغاشمة",
            "rec_en":  "Lock telemetry interface. Rotate authentication keys immediately.",
            "rec_ar":  "قفل واجهة القياس عن بُعد. تدوير مفاتيح المصادقة فورًا.",
        },
        "Replay": {
            "en_name": "Replay Attack",
            "ar_name": "هجوم إعادة التشغيل",
            "rec_en":  "Enable nonce-based anti-replay. Resync timestamps with ground station.",
            "rec_ar":  "تفعيل الحماية من إعادة التشغيل. إعادة مزامنة الطوابع الزمنية.",
        },
        "Unauth": {
            "en_name": "Unauthorised Access",
            "ar_name": "وصول غير مصرح به",
            "rec_en":  "Deny connection. Log endpoint for investigation.",
            "rec_ar":  "رفض الاتصال. تسجيل نقطة النهاية للتحقيق.",
        },
        "BENIGN": {
            "en_name": "Normal Traffic",
            "ar_name": "حركة مرور طبيعية",
            "rec_en":  "Continue standard monitoring.",
            "rec_ar":  "متابعة المراقبة القياسية.",
        },
    }

    def __init__(self, language: str = "en"):
        if language not in ("en", "ar"):
            raise ValueError("language must be 'en' or 'ar'")
        self.language = language

    def generate_briefing(
        self,
        alert_type: str,
        confidence: float,
        top_features: List[str],
        unit_id: str = DEPLOYMENT_UNIT,
    ) -> str:
        """
        Generate a tactical narrative report.

        Parameters
        ----------
        alert_type    : attack category label (e.g. 'DoS', 'BENIGN')
        confidence    : model confidence score (0.0 – 1.0)
        top_features  : list of top SHAP feature names (Truth Triggers)
        unit_id       : drone / unit identifier string
        """
        meta   = self.ATTACK_META.get(alert_type, self.ATTACK_META["Unauth"])
        is_ar  = self.language == "ar"

        name   = meta["ar_name"] if is_ar else meta["en_name"]
        rec    = meta["rec_ar"]  if is_ar else meta["rec_en"]
        feats  = "، ".join(top_features) if is_ar else ", ".join(top_features)
        region = REGION

        if is_ar:
            lines = [
                "━" * 55,
                f"  [A-DIDS] تنبيه أمني تكتيكي — {region}",
                "━" * 55,
                f"  نوع الهجوم     : {name}",
                f"  درجة الثقة     : {confidence:.2%}",
                f"  الوحدة         : {unit_id}",
                f"  مؤشرات الكشف   : {feats}",
                f"  الإجراء الموصى : {rec}",
                "━" * 55,
            ]
        else:
            lines = [
                "━" * 55,
                f"  [A-DIDS] TACTICAL SECURITY ALERT — {region}",
                "━" * 55,
                f"  Alert Type   : {name}",
                f"  Confidence   : {confidence:.2%}",
                f"  Unit         : {unit_id}",
                f"  Truth Triggers: {feats}",
                f"  Action       : {rec}",
                "━" * 55,
            ]
        return "\n".join(lines)


if __name__ == "__main__":
    brief = TacticalBriefing("en")
    print(brief.generate_briefing("DoS", 0.998, ["Entropy", "Rate", "Duration"]))
    print()
    brief_ar = TacticalBriefing("ar")
    print(brief_ar.generate_briefing("DoS", 0.998, ["Entropy", "Rate", "Duration"]))
