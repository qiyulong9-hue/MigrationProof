import argparse
import json
import re
import time
from pathlib import Path

import torch
import torch_npu
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.utils import logging

from migrationproof.p0_model_output import parse_model_output


JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)
VALID_STATUS = {"pass", "conditional_pass", "fail", "insufficient_evidence"}


def normalize(messages):
    result = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        result.append({"role": message["role"], "content": content})
    return result


def parse_legacy(text):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--raw-output", required=True, type=Path)
    args = parser.parse_args()
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.open("x").close()
    args.raw_output.open("x").close()
    logging.set_verbosity_error()
    logging.disable_progress_bar()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager")
    model.eval().to("npu:0")
    requests = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    with args.predictions.open("a", encoding="utf-8", newline="\n") as predictions, args.raw_output.open("a", encoding="utf-8", newline="\n") as raw:
        for index, request in enumerate(requests, 1):
            inputs = processor.apply_chat_template(normalize(request["messages"]), tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt")
            inputs = {key: value.to("npu:0") if hasattr(value, "to") else value for key, value in inputs.items()}
            torch.npu.synchronize(); started = time.perf_counter()
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=request.get("generation", {}).get("max_new_tokens", 160), do_sample=False)
            torch.npu.synchronize(); elapsed = time.perf_counter() - started
            text = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            result = parse_model_output(text)
            predictions.write(json.dumps({"id": request["id"], **result}, ensure_ascii=False) + "\n")
            raw.write(json.dumps({"id": request["id"], "elapsed_seconds": elapsed, "raw_text": text}, ensure_ascii=False) + "\n")
            predictions.flush(); raw.flush()
            print(json.dumps({"progress": f"{index}/{len(requests)}", "id": request["id"], "status": result["status"], "seconds": round(elapsed, 3)}), flush=True)


if __name__ == "__main__":
    main()
