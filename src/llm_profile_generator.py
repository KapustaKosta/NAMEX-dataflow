from __future__ import annotations

import json
import os
import re
import yaml
from typing import Any, Dict, List, Optional
import pandas as pd

class LLMProfileGenerator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            try:
                import streamlit as st
                self.api_key = st.secrets.get("OPENAI_API_KEY")
            except Exception:
                pass
        
    def generate_profile(
        self,
        document_context: Dict[str, Any],
        user_instruction: str,
        existing_schema: Optional[Dict[str, Any]] = None,
        model: str = "gpt-4o",  # Defaulting to a strong model
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
            
        prompt = self._build_prompt(document_context, user_instruction, existing_schema)
        
        # In a real implementation, we would call OpenAI API here.
        # For now, I'll implement a helper that we can mock in tests.
        response_text = self._call_openai(prompt, model)
        
        try:
            # The LLM might return JSON or YAML. We prefer JSON in the response for reliability.
            # But the user wants a YAML profile file.
            profile_data = self._parse_llm_response(response_text)
            return profile_data
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}\nResponse: {response_text}")

    def _build_prompt(
        self,
        document_context: Dict[str, Any],
        user_instruction: str,
        existing_schema: Optional[Dict[str, Any]] = None
    ) -> str:
        # Build context from OCR rows/candidates
        ocr_context = self._format_document_context(document_context)
        
        schema_sample = ""
        if existing_schema:
            schema_sample = f"\nExample of existing profile schema:\n{json.dumps(existing_schema, indent=2)}\n"

        prompt = f"""
You are an expert at creating parsing profiles for a data extraction system.
Your task is to generate a YAML parsing profile based on the user's natural language instruction and the provided document context.

USER INSTRUCTION:
{user_instruction}

DOCUMENT CONTEXT (OCR Rows/Candidates):
{ocr_context}
{schema_sample}

REQUIREMENTS:
1. Output ONLY a valid YAML object that matches the application's profile schema.
2. Do not include markdown formatting (no ```yaml).
3. Do not include explanations.
4. Use robust semantic selectors (row matching rules) instead of brittle row IDs when possible.
5. If the user asks to exclude something, include exclusion rules.
6. The profile must include: profile_name, display_name, extraction (source, ocr engine), and blocks/tables configuration.
7. Use 'yandex_vision' as the default OCR engine if OCR is needed.

PROFILE SCHEMA HINTS:
- Use 'row_selector' with 'include' and 'exclude' if needed.
- Supported inclusion/exclusion types: 'contains', 'code_prefix', 'manual_selected_rows'.
- Supported reconstruction methods: 'pair_name_row_with_following_value_row', 'none'.
- Column mapping should define roles: 'code', 'name', 'unit', 'value'.

Example output structure:
profile_name: my_profile
display_name: My Custom Profile
extraction:
  source: ocr
  ocr:
    engine: yandex_vision
    pages: [2]
blocks:
  - selector:
      block_uids: ["ocr_candidate:2:ocr_p2_fallback"]
    row_filters:
      include:
        any:
          - contains: "Target text"
      exclude:
        any:
          - contains: "Exclude this"
    column_mapping:
      column_1: {{role: "name"}}
      column_2: {{role: "value", value_type: "numeric"}}
"""
        return prompt

    def _format_document_context(self, document_context: Dict[str, Any]) -> str:
        pages_context = []
        ocr_candidates_df = document_context.get("ocr_candidates_df")
        if ocr_candidates_df is not None and not ocr_candidates_df.empty:
            for _, row in ocr_candidates_df.iterrows():
                page = row.get("page")
                block_id = row.get("ocr_block_id")
                text = row.get("evidence_text") or row.get("block_text") or ""
                # Provide a sample of rows if possible
                rows_sample = []
                if "rows" in row and isinstance(row["rows"], list):
                    for r in row["rows"][:20]: # Limit context size
                        rows_sample.append(f"  row {r.get('row_index')}: {r.get('text')}")
                
                block_context = f"Page {page}, Block {block_id}:\n" + "\n".join(rows_sample)
                pages_context.append(block_context)
        
        return "\n\n".join(pages_context)

    def _call_openai(self, prompt: str, model: str) -> str:
        # This will be mocked in tests. In production, it uses the openai library.
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {{"role": "system", "content": "You are a helpful assistant that generates YAML profiles."}},
                    {{"role": "user", "content": prompt}}
                ],
                temperature=0,
            )
            return response.choices[0].message.content or ""
        except ImportError:
            return "# openai library not installed\n{}"
        except Exception as e:
            return f"# Error calling OpenAI: {e}\n{{}}"

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        # Strip markdown if LLM ignored instructions
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:yaml|json)?\n", "", cleaned)
            cleaned = re.sub(r"\n```$", "", cleaned)
        
        # Try YAML first
        try:
            return yaml.safe_load(cleaned)
        except Exception:
            # Try JSON if YAML fails
            return json.loads(cleaned)

def validate_generated_profile(profile_data: Dict[str, Any]) -> List[str]:
    errors = []
    if not profile_data.get("profile_name"):
        errors.append("Missing profile_name")
    if not profile_data.get("display_name"):
        errors.append("Missing display_name")
    
    extraction = profile_data.get("extraction", {})
    if not extraction.get("source"):
        errors.append("Missing extraction.source")
    
    blocks = profile_data.get("blocks") or profile_data.get("tables")
    if not blocks:
        errors.append("Missing blocks or tables configuration")
    
    return errors
