import re
import unicodedata

class ArabicPostProcessor:

    def process(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        text = self._normalize_arabic_chars(text)
        text = self._remove_diacritics(text)
        text = self._normalize_whitespace(text)
        text = self._clean_noise(text)

        return text.strip()

    def _normalize_arabic_chars(self, text: str) -> str:
        replacements = {
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',
            'ى': 'ي',
            'ؤ': 'و',
            'ئ': 'ي'
        }

        for k, v in replacements.items():
            text = text.replace(k, v)

        return text

    def _remove_diacritics(self, text: str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFC', text)
            if unicodedata.category(c) != 'Mn'
        )

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def _clean_noise(self, text: str) -> str:
        # remove tatweel
        text = text.replace('ـ', '')

        # remove standalone numbers lines (but keep numbers inside text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        # DO NOT remove non-arabic chars (important for RAG)
        return text