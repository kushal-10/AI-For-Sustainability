#!/usr/bin/env python3
# Prompts + object builders. No I/O here.

# -------------------- SYSTEM PROMPTS (short, one mixed example each) --------------------

SYS_PROMPT_SDG = r"""
You analyze corporate disclosures for SDG mentions.
Input:
- Passage (plain text).
- SDG_HITS: JSON with lists of REGEX PATTERNS that ALREADY MATCHED the passage.
Do NOT match yourself.

Task:
For EACH regex pattern in SDG_HITS, classify the passage’s mention as "symbolic" or "substantive".

Definitions:
- symbolic: vague/aspirational, policy or compliance-only (e.g., legal citations), generic intentions without concrete actions, resources, timelines, KPIs, or measured outcomes.
- substantive: concrete implementation/results (projects, pilots, budgets/teams, timelines, KPIs/metrics, verified/assured reporting, quantified impact).

Output:
- ONLY a JSON/dict
- KEYS: the EXACT regex strings (preserve escaping)
- VALUES: "symbolic" or "substantive"
- No explanations or extra fields.

Example (one symbolic, one substantive):
Input SDG_HITS:
{
  "hits_sdg13": ["\\benvironmental\\s+impact\\b", "\\bgreenhouse\\s+gas\\w*\\b"]
}
Passage:
"We aim to reduce our environmental impact. In 2024 we cut Scope 2 greenhouse gases by 12% via PPAs (assured)."
Expected output:
{
  "\\benvironmental\\s+impact\\b": "symbolic",
  "\\bgreenhouse\\s+gas\\w*\\b": "substantive"
}
"""

SYS_PROMPT_TECH = r"""
You analyze corporate disclosures for technology mentions.
Input:
- Passage (plain text).
- TECH_HITS: JSON with lists of REGEX PATTERNS that ALREADY MATCHED the passage.
Do NOT match yourself.

Task:
For EACH regex pattern in TECH_HITS, classify the passage’s mention as "symbolic" or "substantive".

Definitions:
- symbolic: vague/aspirational/strategy-only; no concrete implementation or measured outcomes.
- substantive: concrete use or results (named systems/tools, pilots/production, KPIs/metrics, budgets/teams, timelines, specific integrations/vendors, audits/assurance).

Output:
- ONLY a JSON/dict
- KEYS: the EXACT regex strings (preserve escaping)
- VALUES: "symbolic" or "substantive"
- No explanations or extra fields.

Example (one symbolic, one substantive):
Input TECH_HITS:
{
  "hits_ai_ml": ["\\bartificial\\s+intelligence\\b"],
  "hits_cloud_computing": ["\\bcloud\\s+computing\\b"]
}
Passage:
"We plan to explore artificial intelligence next year. 40% of data workloads now run in production on cloud computing with monitored SLAs."
Expected output:
{
  "\\bartificial\\s+intelligence\\b": "symbolic",
  "\\bcloud\\s+computing\\b": "substantive"
}
"""

# -------------------- BATCH OBJECT BUILDERS --------------------

def create_batch_object_sdg(passage: str, global_id: str, sdg_hits: dict, model: str = "gpt-4.1-mini"):
    user_content = f"Passage:\n{passage}\n\nSDG_HITS (regex patterns already matched):\n{sdg_hits}"
    return {
        "custom_id": f"sdg||{global_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": SYS_PROMPT_SDG},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": 150,
            "temperature": 0
        }
    }

def create_batch_object_tech(passage: str, global_id: str, tech_hits: dict, model: str = "gpt-4.1-mini"):
    user_content = f"Passage:\n{passage}\n\nTECH_HITS (regex patterns already matched):\n{tech_hits}"
    return {
        "custom_id": f"tech||{global_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": SYS_PROMPT_TECH},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": 150,
            "temperature": 0
        }
    }
