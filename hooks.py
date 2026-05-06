import re

# Maps the auto-generated English title (capitalized type name) to Croatian.
# Explicit per-callout titles (e.g. !!! warning "Vlastiti naslov") are left untouched.
_HR_TITLES = {
    "Note": "Napomena",
    "Abstract": "Sažetak",
    "Summary": "Sažetak",
    "Tldr": "Sažetak",
    "Info": "Informacija",
    "Todo": "Za napraviti",
    "Tip": "Savjet",
    "Hint": "Savjet",
    "Important": "Važno",
    "Success": "Uspjeh",
    "Check": "Provjera",
    "Done": "Gotovo",
    "Question": "Pitanje",
    "Help": "Pomoć",
    "Faq": "ČPP",
    "Warning": "Upozorenje",
    "Caution": "Oprez",
    "Attention": "Pažnja",
    "Failure": "Neuspjeh",
    "Fail": "Neuspjeh",
    "Missing": "Nedostaje",
    "Danger": "Opasnost",
    "Error": "Greška",
    "Bug": "Programska greška",
    "Example": "Primjer",
    "Quote": "Citat",
    "Cite": "Citat",
}

_TITLE_RE = re.compile(r'<p class="admonition-title">([^<]+)</p>')


def on_page_content(html, page):
    if not page.file.src_path.startswith("hr/"):
        return html

    def _replace(m):
        title = m.group(1)
        return f'<p class="admonition-title">{_HR_TITLES.get(title, title)}</p>'

    return _TITLE_RE.sub(_replace, html)
