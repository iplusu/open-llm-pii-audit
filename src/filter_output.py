"""Output filter (conditions C3/C5/C6/C8): mask PII patterns in model output.

Moved unchanged from the tested pii_lib.apply_output_filter. Pure stdlib.
Order matters: card, email, address, phone, ssn, then names (cascade-aware, first name dropped, surname kept).
"""
from __future__ import annotations

import re

FILTER_TOKENS = {
    "ssn": "[FILTERED_SSN]",
    "phone": "[FILTERED_PHONE]",
    "card": "[FILTERED_CARD]",
    "email": "[FILTERED_EMAIL]",
    "address": "[ADDRESS]",
    "name": "",  # name filtering is handled separately (see filter_output)
}


_PATTERNS: dict[str, re.Pattern] = {
    # SSN: 123-45-6789 / 123.45.6789 / 123456789 (9 digits, no lookbehind digit)
    "ssn": re.compile(r"(?<!\d)\d{3}[-.]?\d{2}[-.]?\d{4}(?!\d)"),
    # phone: (555) 014-2830 / 555-014-2830 / 555 014 2830 / +1 555 014 2830
    "phone": re.compile(
        r"(?<!\d)(?:\+?1[\s.-]?)?\(?555\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"
    ),
    # card: 4000 0000 1111 2222 (4x4 groups, allow dashes)
    "card": re.compile(r"(?<!\d)(?:\d{4}[\s-]){3}\d{4}(?!\d)"),
    # email: full address on the synthetic domains used by make_probes.py
    "email": re.compile(
        r"[A-Za-z0-9._%+-]+@(?:syntheticmail\.io|example\.pii|piitest\.org)"
    ),
    # address: "1234 Maple St, Springdale, OH 44807"
    "address": re.compile(
        r"\d{1,5}\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\s+"
        r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Ct|Court)\b"
        r"(?:,\s*[A-Za-z .'-]+)?"
        r"(?:,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?"
    ),
    # name: two capitalized words, neither in the stop-list
    "name": re.compile(
        r"\b([A-Z][a-z]{1,20})\s+([A-Z][a-z]{1,20})\b"
    ),
}


_NAME_STOPWORDS = {
    "Please", "Hello", "Hi", "Hey", "Can", "Could", "Would", "What", "Which",
    "Who", "When", "Where", "Why", "How", "I", "You", "We", "They", "She",
    "He", "It", "My", "Your", "The", "This", "That", "These", "Those",
    "Now", "Then", "There", "Here", "Today", "Yesterday", "Tomorrow",
    "Morning", "Afternoon", "Evening", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday", "January", "February",
    "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December", "Dear", "Sincerely", "Regards",
    "Best", "Thanks", "Thank", "Note", "Subject", "Re", "From", "To",
    "Date", "Order", "Package", "Delivery", "Shipping", "Invoice",
    "Payment", "Account", "Card", "Phone", "Email", "Address", "Social",
    "Security", "Number", "Name", "Customer", "Support", "Assistant",
    "Doctor", "Patient", "Clinic", "Hospital", "Report", "Summary",
    "Please", "Ignore", "Previous", "Instructions", "New", "Task",
    "Role", "Play", "As", "A", "An", "Of", "And", "For", "With", "Without",
    "At", "In", "On", "By", "About", "Into", "Over", "After", "Before",
    "Springdale", "Riverton", "Ashford", "Lakemont", "Brookfield", "Fairview",
    "Oakdale", "Cedarville", "Maple", "Cedar", "Oak", "Pine", "Elm",
    "Birch", "Willow", "Magnolia", "Chestnut", "Sycamore", "Hickory",
    "Washington", "Lincoln", "Jefferson", "Madison", "Franklin", "Adams",
    "Monroe", "Jackson", "Polk", "Taylor", "Ohio", "Kentucky", "Indiana",
    "Pennsylvania", "Michigan", "Illinois", "Wisconsin", "Minnesota",
    "Georgia", "Virginia", "Carolina", "Delaware", "Massachusetts",
    "Connecticut", "Maine", "New", "Jersey", "York", "Hampshire",
    "Rhode", "Island", "Vermont", "North", "South", "East", "West",
}


def apply_output_filter(text: str) -> tuple[str, int]:
    """Mask PII patterns in model output. Returns (filtered_text, n_removed)."""
    n_removed = 0

    def _sub(pat, token):
        nonlocal n_removed
        text_out, k = pat.subn(token, text)
        n_removed += k
        return text_out

    # Order matters: card before ssn/phone (card is 4x4 digits; ssn is 9),
    # email before address (emails contain no street pattern), then address,
    # phone, ssn, and finally names.
    text = _sub(_PATTERNS["card"], FILTER_TOKENS["card"])
    text = _sub(_PATTERNS["email"], FILTER_TOKENS["email"])
    text = _sub(_PATTERNS["address"], FILTER_TOKENS["address"])
    text = _sub(_PATTERNS["phone"], FILTER_TOKENS["phone"])
    text = _sub(_PATTERNS["ssn"], FILTER_TOKENS["ssn"])
    # Names: two-capital-word pairs not in the stoplist, and not inside an
    # already-filtered token. The scan is cascade-aware: after a firing pair
    # (first name dropped, surname kept), the cursor re-examines FROM the kept
    # surname so an adjacent pair ("Pia Hartwell" after "Contact") is still
    # tested; after a stopword-skipped pair the cursor advances one character
    # so the adjacent pair is still tested. A fired pair keeps the surname.
    name_pat = re.compile(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b")
    i = 0
    while i < len(text):
        m = name_pat.search(text, i)
        if not m:
            break
        a, b = m.group(1), m.group(2)
        if a in _NAME_STOPWORDS or b in _NAME_STOPWORDS:
            i = m.start() + 1
        else:
            text = text[: m.start()] + b + text[m.end():]
            n_removed += 1
            i = m.start()  # cascade: re-examine from the kept surname
    return text, n_removed
