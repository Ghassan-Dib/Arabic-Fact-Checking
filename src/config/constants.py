WHITELISTED_SITES = [
    "misbar.com",
    "yoopyup.com",
    "fatabyyano.net",
    "almushahid.net",
    "annir.ly",
    "falso.ly",
    "akhbarmeter.org",
]

LABELS_MAP = {
    "refuted": [
        "خطأ",
        "خاطئ",
        "كاذب",
        "خبر كاذب",
        "غير صحيح",
        "مش حقيقي",
        "الادعاء خاطئ",
        "زائف",
        "مزيف",
        "وليدة ذكاء إصطناعي",
        "fake",
        "اشاعة",
        "مفبرك",
        "مضلل",
        "مضلّل",
        "misleading",
        "فيديو معدّل",
        "معدّل",
        "معدل",
    ],
    "supported": ["صحيح", "correct", "نص حقيقي"],
    "Not Enough Evidence": [
        "محتوى ناقص",
        "سياق ناقص",
    ],
    "Conflicting Evidence/Cherrypicking": [
        "زائف جزئيًا",
        "زائف جزئي",
        "غير صحيح جزئياً",
        "غير صحيح جزئيا",
        "صحيح جزئيًا",
        "صحيح جزئياً",
        "خاطئ جزئيا",
        "خطأ جزئياً",
        "نصف حقيقي",
    ],
}

REMOVAL_KEYWORDS = [
    "مواضيع أخرى قد تهمك",
    "Related topics",
    "اقرأ/ي أيضًا",
    "اقرأ أيضاً",
    "قد يهمك",
    "مصادر الادعاء",
    "مصادر الادعاء:",
    "رابط الادعاء:",
    "المصادر",
    "Topic categories",
    "Claim sources",
]

VERDICTS = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]
