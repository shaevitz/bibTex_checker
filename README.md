# bibtex_checker

CLI tool to validate and auto-correct BibTeX metadata.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Usage

```bash
python bibtex_checker.py input.bib
python bibtex_checker.py input.bib --fix
python bibtex_checker.py input.bib --fix --report custom_report.md --out-bib custom_corrected.bib
```

## Behavior

- Validates required fields per BibTeX type.
- Validates and normalizes key metadata fields (`author`, `title`, `journal`, `year`, `volume`, `number`, `pages`, `doi`, `url`, and `booktitle` where needed).
- Verification source order:
  1. DOI
  2. URL metadata
  3. Crossref search
  4. Semantic Scholar search
  5. Google fallback
- Writes Markdown report to `<input>_report.md` by default.
- With `--fix`, writes corrected BibTeX to `<input>_corrected.bib` by default.

## Exit codes

- `0`: completed with no unresolved entries
- `1`: completed with unresolved entries
- `2`: fatal error (for example parse failure or missing input)

## Notes

- Citation keys are never renamed.
- Google fallback is best-effort and may be less reliable than DOI/Crossref/Semantic Scholar.
- The program applies all deterministic fixes it can and reports unresolved items in both markdown and terminal output.
