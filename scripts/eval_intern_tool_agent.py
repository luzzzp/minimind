import argparse
import json
import os
import re
import sys
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.train_agent import execute_tool, parse_tool_calls, validate_gt_in_text
from trainer.trainer_utils import get_model_params, setup_seed

warnings.filterwarnings("ignore")


def load_jsonl(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from, trust_remote_code=True)
    if "model" in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
        ))
        moe_suffix = "_moe" if args.use_moe else ""
        ckp = os.path.join(args.save_dir, f"{args.weight}_{args.hidden_size}{moe_suffix}.pth")
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def generate_once(model, tokenizer, messages, tools, args):
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
        open_thinking=bool(args.open_thinking),
    )
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False, truncation=True).to(args.device)
    start = time.time()
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=bool(args.do_sample),
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, gen_tokens, time.time() - start


def normalize_args(args):
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {}
    return args if isinstance(args, dict) else {}


def args_match(pred, expected):
    pred, expected = normalize_args(pred), normalize_args(expected)
    if not expected:
        return True
    ok = 0
    for key, value in expected.items():
        if key not in pred:
            continue
        if isinstance(value, (int, float)):
            try:
                ok += abs(float(pred[key]) - float(value)) < 1e-6
            except Exception:
                pass
        else:
            ok += str(pred[key]).lower() == str(value).lower()
    return ok == len(expected)


def has_bad_tool_format(text):
    return text.count("<tool_call>") != text.count("</tool_call>") or bool(re.search(r"<tool_call>\s*</tool_call>", text, re.DOTALL))


def run_case(row, model, tokenizer, args):
    messages = [{"role": "user", "content": row["prompt"]}]
    all_calls = []
    outputs = []
    total_tokens = 0
    total_time = 0.0
    unfinished = False

    for turn in range(args.max_turns):
        text, gen_tokens, spent = generate_once(model, tokenizer, messages, row["tools"], args)
        outputs.append(text)
        total_tokens += gen_tokens
        total_time += spent
        calls = parse_tool_calls(text)
        all_calls.extend(calls)
        if not calls:
            break
        unfinished = turn == args.max_turns - 1
        messages.append({"role": "assistant", "content": text})
        for call in calls:
            name = call.get("name", "")
            raw = normalize_args(call.get("arguments", {}))
            result = execute_tool(name, raw)
            messages.append({"role": "tool", "content": json.dumps(result or {"error": "tool failed"}, ensure_ascii=False)})

    final_text = outputs[-1] if outputs else ""
    expected_tools = row.get("expected_tools", [])
    expected_args = row.get("expected_arguments", [])
    pred_tools = [call.get("name", "") for call in all_calls]
    valid_json = sum(1 for call in all_calls if isinstance(call, dict) and call.get("name"))
    tool_name_ok = pred_tools[:len(expected_tools)] == expected_tools
    arg_ok = len(all_calls) >= len(expected_args) and all(args_match(call.get("arguments", {}), exp) for call, exp in zip(all_calls, expected_args))
    answer_hits = validate_gt_in_text(final_text, row.get("gt", []))
    answer_ok = len(answer_hits) == len(row.get("gt", [])) if row.get("gt") else bool(final_text.strip())
    format_bad = any(has_bad_tool_format(x) for x in outputs)
    success = bool(tool_name_ok and arg_ok and answer_ok and not unfinished and not format_bad)

    return {
        "json_valid": int(valid_json == len(all_calls) and len(all_calls) > 0),
        "tool_name_ok": int(tool_name_ok),
        "arg_ok": int(arg_ok),
        "answer_ok": int(answer_ok),
        "format_bad": int(format_bad),
        "success": int(success),
        "gen_tokens": total_tokens,
        "seconds": total_time,
        "avg_len": len(final_text),
    }


def summarize(metrics):
    n = max(len(metrics), 1)
    fields = ["json_valid", "tool_name_ok", "arg_ok", "answer_ok", "success", "format_bad"]
    out = {field: sum(m[field] for m in metrics) / n for field in fields}
    tokens = sum(m["gen_tokens"] for m in metrics)
    seconds = sum(m["seconds"] for m in metrics)
    out["avg_response_chars"] = sum(m["avg_len"] for m in metrics) / n
    out["tokens_per_second"] = tokens / seconds if seconds > 0 else 0.0
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate MiniMind-ToolRL held-out tool tasks.")
    parser.add_argument("--eval_path", default="../dataset/intern_tool_eval.jsonl", type=str)
    parser.add_argument("--load_from", default="../model", type=str)
    parser.add_argument("--save_dir", default="../out", type=str)
    parser.add_argument("--weight", default="intern_grpo", type=str)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--max_new_tokens", default=384, type=int)
    parser.add_argument("--max_turns", default=3, type=int)
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--do_sample", default=0, type=int)
    parser.add_argument("--open_thinking", default=0, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    args = parser.parse_args()

    setup_seed(args.seed)
    model, tokenizer = init_model(args)
    rows = load_jsonl(args.eval_path, limit=args.limit or None)
    metrics = [run_case(row, model, tokenizer, args) for row in rows]
    summary = summarize(metrics)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
