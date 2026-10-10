"""Licence record validation. A dataset whose licence cannot be established is FLAGGED for review, never assumed free."""
import sys

REQUIRED = ("name", "license", "license_url", "provider", "restrictions")


def validate(record):
    problems = [f"missing field: {k}" for k in REQUIRED if not record.get(k)]
    text = str(record.get("license", "")).lower()
    needs_review = bool(problems) or any(w in text for w in ("unknown", "unspecified", "tbd", "check"))
    return {"valid": not problems, "needs_review": needs_review, "problems": problems}


if __name__ == "__main__":
    import json
    print(json.dumps(validate(json.loads(open(sys.argv[1]).read())), indent=1))
