import json
import re


JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)
VALID_STATUS = {"pass", "conditional_pass", "fail", "insufficient_evidence"}


def parse_model_output(text):
    for match in JSON_OBJECT.finditer(text):
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        status = payload.get("status")
        issues = payload.get("issues")
        if status in VALID_STATUS and isinstance(issues, list):
            return {"status": status, "issues": issues, "rationale": str(payload.get("rationale", ""))[:1000]}
    return {"status": "insufficient_evidence", "issues": ["invalid_json_output"], "rationale": text[:1000]}
