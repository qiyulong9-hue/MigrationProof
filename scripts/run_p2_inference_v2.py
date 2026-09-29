import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu
from transformers import AutoModelForImageTextToText, AutoProcessor, StoppingCriteria, StoppingCriteriaList
from transformers.utils import logging

from migrationproof.p2_output_contract import OutputContractError, parse_model_output


def normalize(messages):
    return [{"role": item["role"], "content": [{"type": "text", "text": item["content"]}]} for item in messages]


class StopAfterValidJson(StoppingCriteria):
    def __init__(self, processor, prompt_length):
        self.processor = processor
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        text = self.processor.batch_decode(input_ids[:, self.prompt_length:], skip_special_tokens=True)[0].strip()
        try:
            parse_model_output(text)
            return True
        except OutputContractError:
            return False


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--input", required=True, type=Path)
parser.add_argument("--predictions", required=True, type=Path)
parser.add_argument("--raw-output", required=True, type=Path)
args = parser.parse_args()
args.predictions.parent.mkdir(parents=True, exist_ok=True)
args.predictions.open("x").close(); args.raw_output.open("x").close()
logging.set_verbosity_error(); logging.disable_progress_bar()
processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager").eval().to("npu:0")
requests = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
with args.predictions.open("a", encoding="utf-8", newline="\n") as predictions, args.raw_output.open("a", encoding="utf-8", newline="\n") as raw:
    for index, request in enumerate(requests, 1):
        inputs = processor.apply_chat_template(normalize(request["messages"]), tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = {key: value.to("npu:0") if hasattr(value, "to") else value for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        stopping = StoppingCriteriaList([StopAfterValidJson(processor, prompt_length)])
        torch.npu.synchronize(); started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=request["generation"]["max_new_tokens"], do_sample=False, stopping_criteria=stopping)
        torch.npu.synchronize(); elapsed = time.perf_counter() - started
        text = processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0].strip()
        try:
            parsed = parse_model_output(text)
            result = {"decision": parsed.decision, "issue_type": parsed.issue_type, "evidence": list(parsed.evidence), "explanation": parsed.explanation, "valid_output": True}
        except OutputContractError:
            result = {"decision": "fail", "issue_type": "unsafe_pass", "evidence": ["invalid_structured_output"], "explanation": "Output contract violation.", "valid_output": False}
        predictions.write(json.dumps({"id": request["id"], **result}, ensure_ascii=False) + "\n"); predictions.flush()
        raw.write(json.dumps({"id": request["id"], "elapsed_seconds": elapsed, "raw_text": text}, ensure_ascii=False) + "\n"); raw.flush()
        print(json.dumps({"progress": f"{index}/{len(requests)}", "id": request["id"], "valid": result["valid_output"], "seconds": round(elapsed, 3)}), flush=True)
