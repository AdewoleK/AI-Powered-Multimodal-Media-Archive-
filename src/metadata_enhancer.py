import re

YEAR_PATTERN = re.compile(r'\b(19\d{2}|20\d{2})\b')

def extract_years(text: str):
    """Extract unique 4-digit years (1900–2099) from text"""
    matches = YEAR_PATTERN.findall(text)
    return sorted(set(int(year) for year in matches))