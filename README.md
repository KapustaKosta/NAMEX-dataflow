# NAMEX DataFlow

Streamlit MVP for external document processing. The app accepts a user document,
extracts raw rows, normalizes common fields, validates typical data quality
issues, allows manual review in a table editor, and exports the final clean
table to Excel.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Supported Formats

- CSV
- XLSX
- PDF

## Поддерживаемые профили документов

- `generic_pdf`: базовое извлечение текста и таблиц из PDF.
- `fish_market_report`: отчет "Рыбные ряды" / НАЦРЫБРЕСУРС,
  извлечение таблиц улова, оптовых цен, мировых ценовых индикаторов
  и розничных цен на мороженую рыбу.

## Project Flow

```text
upload -> extraction -> normalization -> validation -> human review -> export
```

All extractors return one standard pandas DataFrame with audit fields such as
`source_file`, `page`, `sheet`, and `evidence_text`.

## User Profile Builder

The Streamlit app includes a no-code source profile builder for generic PDFs.
Users can select extracted PDF tables or OCR candidates, assign column roles,
configure simple row filters, preview structured rows, save the profile under
`profiles/user_profiles/`, and apply it to similar documents without adding a
new hardcoded Python parser.

## MVP Limits

- CSV and XLSX extraction uses simple pandas parsing and lightweight column
  matching.
- PDF extraction uses `pdfplumber` to extract tables or page text fragments.
- OCR is a placeholder for scanned PDFs.
- LLM extraction is a placeholder and does not call any external API.
- The MVP does not use a database or background workers.

## OCR and LLM Extension

Add OCR in `src/extract_ocr.py` for scanned PDFs that do not contain embedded
text. Add LLM extraction in `src/extract_llm.py` only for cases where rules,
tables, and simple parsers cannot reliably extract structured fields. Both
modules should keep returning the standard DataFrame schema with evidence for
auditability.
