import re

class EnglishPostProcessor:

    def process(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        text = self._fix_encoding_glitches(text)
        text = self._normalize_whitespace(text)
        text = self._clean_punctuation_spacing(text)
        text = self._remove_noise_lines(text)

        return text.strip()

    def _fix_encoding_glitches(self, text: str) -> str:
        replacements = {
            '\u2018': "'", '\u2019': "'",
            '\u201c': '"', '\u201d': '"',
            '\u2013': '-', '\u2014': '--',
            '\u00a0': ' ',
            '\u2026': '...',
            '\ufeff': '',
            '\u200b': ''
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def _clean_punctuation_spacing(self, text: str) -> str:
        return re.sub(r'\s+([.,;:!?])', r'\1', text)

    def _remove_noise_lines(self, text: str) -> str:
        # remove standalone numbers only lines
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        # remove excessive empty lines at edges
        return text