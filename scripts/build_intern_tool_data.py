import argparse
import json
import math
import os
import random
from copy import deepcopy


TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
]

TOOL_MAP = {tool["function"]["name"]: tool for tool in TOOLS}

WEATHER_DATA = {
    "北京": ("28°C", "晴"),
    "上海": ("15°C", "多云"),
    "广州": ("32°C", "闷热"),
    "深圳": ("30°C", "晴"),
    "杭州": ("22°C", "阴"),
    "成都": ("18°C", "小雨"),
    "Tokyo": ("12°C", "晴"),
    "New York": ("8°C", "多云"),
}
TIME_DATA = {
    "Asia/Shanghai": "2025-03-07 14:30:00",
    "America/New_York": "2025-03-07 01:30:00",
    "Europe/London": "2025-03-07 06:30:00",
    "Asia/Tokyo": "2025-03-07 15:30:00",
}
EXCHANGE_DATA = {
    ("USD", "CNY"): 7.21,
    ("EUR", "CNY"): 7.85,
    ("GBP", "CNY"): 9.12,
    ("JPY", "CNY"): 0.048,
    ("USD", "EUR"): 0.92,
    ("CNY", "JPY"): 20.83,
}
TRANSLATE_DATA = {
    ("你好世界", "english"): "Hello World",
    ("早上好", "english"): "Good morning",
    ("Good morning", "chinese"): "早上好",
    ("今天天气真好", "english"): "The weather is nice today",
    ("I love programming", "chinese"): "我喜欢编程",
    ("机器学习很有趣", "english"): "Machine learning is interesting",
}
UNIT_DATA = {
    ("km", "miles"): 0.621371,
    ("miles", "km"): 1.60934,
    ("kg", "pounds"): 2.20462,
    ("pounds", "kg"): 0.453592,
    ("meters", "feet"): 3.28084,
    ("feet", "meters"): 0.3048,
    ("celsius", "fahrenheit"): None,
    ("fahrenheit", "celsius"): None,
}


def dump_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tool_schema(names):
    return [deepcopy(TOOL_MAP[name]) for name in names]


def tool_call(name, arguments):
    return {"name": name, "arguments": arguments}


def tool_call_text(name, arguments):
    return "<tool_call>\n" + json.dumps(tool_call(name, arguments), ensure_ascii=False) + "\n</tool_call>"


def result_for(name, arguments):
    if name == "calculate_math":
        expr = str(arguments["expression"]).replace("^", "**")
        value = eval(expr, {"__builtins__": {}, "math": math})
        return {"result": str(round(value, 6)).rstrip("0").rstrip(".")}
    if name == "unit_converter":
        value = float(arguments["value"])
        src, dst = arguments["from_unit"], arguments["to_unit"]
        if (src, dst) == ("celsius", "fahrenheit"):
            out = value * 1.8 + 32
        elif (src, dst) == ("fahrenheit", "celsius"):
            out = (value - 32) / 1.8
        else:
            out = value * UNIT_DATA[(src, dst)]
        return {"result": round(out, 4), "from": f"{value:g} {src}", "to": dst}
    if name == "get_current_weather":
        temp, cond = WEATHER_DATA[arguments["location"]]
        return {"city": arguments["location"], "temperature": temp, "condition": cond, "humidity": "65%"}
    if name == "get_current_time":
        timezone = arguments.get("timezone", "Asia/Shanghai")
        return {"datetime": TIME_DATA[timezone], "timezone": timezone}
    if name == "get_exchange_rate":
        pair = (arguments["from_currency"], arguments["to_currency"])
        return {"from": pair[0], "to": pair[1], "rate": EXCHANGE_DATA[pair]}
    if name == "translate_text":
        key = (arguments["text"], arguments["target_language"])
        return {"translated_text": TRANSLATE_DATA[key]}
    raise ValueError(f"unknown tool: {name}")


def final_answer(name, arguments, result):
    if name == "calculate_math":
        return f"计算结果是 {result['result']}。"
    if name == "unit_converter":
        return f"{result['from']} 约等于 {result['result']} {arguments['to_unit']}。"
    if name == "get_current_weather":
        return f"{result['city']}当前天气为{result['condition']}，温度{result['temperature']}。"
    if name == "get_current_time":
        return f"{result['timezone']} 当前时间是 {result['datetime']}。"
    if name == "get_exchange_rate":
        return f"{result['from']} 兑 {result['to']} 的汇率是 {result['rate']}。"
    if name == "translate_text":
        return f"翻译结果是：{result['translated_text']}。"
    raise ValueError(f"unknown tool: {name}")


def sample_task(rng):
    builders = [
        build_math_task,
        build_unit_task,
        build_weather_task,
        build_time_task,
        build_exchange_task,
        build_translate_task,
        build_two_step_task,
    ]
    return rng.choice(builders)(rng)


def build_math_task(rng):
    a, b = rng.randint(12, 900), rng.randint(3, 99)
    op = rng.choice(["+", "-", "*"])
    expr = f"{a}{op}{b}"
    prompt = rng.choice([f"帮我计算 {a} {op} {b} 的结果", f"{expr} 等于多少？", f"请调用工具算一下 {expr}"])
    return make_single("calculate_math", {"expression": expr}, prompt, ["calculate_math", "get_current_time"])


def build_unit_task(rng):
    src, dst = rng.choice(list(UNIT_DATA.keys()))
    value = rng.choice([10, 25, 30, 50, 100, 128])
    prompt = f"帮我把 {value} {src} 换算成 {dst}"
    return make_single("unit_converter", {"value": value, "from_unit": src, "to_unit": dst}, prompt, ["unit_converter", "calculate_math"])


def build_weather_task(rng):
    city = rng.choice(list(WEATHER_DATA.keys()))
    prompt = rng.choice([f"{city}今天天气怎么样？", f"查询一下{city}的天气", f"What is the weather in {city}?"])
    return make_single("get_current_weather", {"location": city}, prompt, ["get_current_weather", "get_current_time"])


def build_time_task(rng):
    timezone = rng.choice(list(TIME_DATA.keys()))
    prompt = rng.choice([f"现在 {timezone} 是几点？", f"帮我查一下 {timezone} 当前时间"])
    return make_single("get_current_time", {"timezone": timezone}, prompt, ["get_current_time", "get_current_weather"])


def build_exchange_task(rng):
    src, dst = rng.choice(list(EXCHANGE_DATA.keys()))
    prompt = rng.choice([f"查一下 {src} 到 {dst} 的汇率", f"{src} 兑 {dst} 现在是多少？"])
    return make_single("get_exchange_rate", {"from_currency": src, "to_currency": dst}, prompt, ["get_exchange_rate", "get_current_time"])


def build_translate_task(rng):
    text, target = rng.choice(list(TRANSLATE_DATA.keys()))
    prompt = rng.choice([f"把“{text}”翻译成{target}", f"Translate '{text}' to {target}."])
    return make_single("translate_text", {"text": text, "target_language": target}, prompt, ["translate_text", "get_current_time"])


def build_two_step_task(rng):
    city = rng.choice(["Tokyo", "北京", "上海"])
    value = rng.choice([30, 50, 100])
    args1 = {"location": city}
    args2 = {"value": value, "from_unit": "km", "to_unit": "miles"}
    res1 = result_for("get_current_weather", args1)
    res2 = result_for("unit_converter", args2)
    prompt = f"查询{city}天气，并把{value}公里换算成英里"
    return {
        "prompt": prompt,
        "tools": tool_schema(["get_current_weather", "unit_converter", "get_current_time"]),
        "calls": [tool_call("get_current_weather", args1), tool_call("unit_converter", args2)],
        "results": [res1, res2],
        "answer": f"{city}当前天气为{res1['condition']}，温度{res1['temperature']}；{value} km 约等于 {res2['result']} miles。",
        "gt": [res1["temperature"], res1["condition"], str(res2["result"])],
    }


def make_single(name, arguments, prompt, available):
    result = result_for(name, arguments)
    answer = final_answer(name, arguments, result)
    gt = [str(v) for v in result.values() if isinstance(v, (str, int, float))]
    return {
        "prompt": prompt,
        "tools": tool_schema(available),
        "calls": [tool_call(name, arguments)],
        "results": [result],
        "answer": answer,
        "gt": gt,
    }


def sft_row(task):
    conversations = [{"role": "system", "content": "你是一个会严格按工具规范解决问题的AI助手。", "tools": json.dumps(task["tools"], ensure_ascii=False)}]
    conversations.append({"role": "user", "content": task["prompt"]})
    for call, result in zip(task["calls"], task["results"]):
        conversations.append({"role": "assistant", "content": tool_call_text(call["name"], call["arguments"])})
        conversations.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
    conversations.append({"role": "assistant", "content": task["answer"]})
    return {"conversations": conversations}


def dpo_row(task, rng):
    chosen = sft_row(task)["conversations"]
    rejected = deepcopy(chosen)
    mode = rng.choice(["wrong_tool", "wrong_args", "direct_answer", "bad_format"])
    if mode == "direct_answer":
        rejected = rejected[:2] + [{"role": "assistant", "content": "我认为答案大概是 42。"}]
    elif mode == "bad_format":
        rejected[2]["content"] = json.dumps(task["calls"][0], ensure_ascii=False)
    elif mode == "wrong_tool":
        wrong = "get_current_time" if task["calls"][0]["name"] != "get_current_time" else "get_current_weather"
        rejected[2]["content"] = tool_call_text(wrong, {})
    else:
        bad_call = deepcopy(task["calls"][0])
        bad_call["arguments"] = {"value": -1}
        rejected[2]["content"] = tool_call_text(bad_call["name"], bad_call["arguments"])
    return {"chosen": chosen, "rejected": rejected}


def agent_row(task):
    return {
        "conversations": [
            {"role": "system", "content": "你是一个会严格按工具规范解决问题的AI助手。", "tools": json.dumps(task["tools"], ensure_ascii=False)},
            {"role": "user", "content": task["prompt"]},
            {"role": "assistant", "content": ""},
        ],
        "gt": task["gt"],
    }


def eval_row(task):
    return {
        "prompt": task["prompt"],
        "tools": task["tools"],
        "expected_tools": [call["name"] for call in task["calls"]],
        "expected_arguments": [call["arguments"] for call in task["calls"]],
        "gt": task["gt"],
        "answer": task["answer"],
    }


def build_rows(size, seed):
    rng = random.Random(seed)
    return [sample_task(rng) for _ in range(size)]


def main():
    parser = argparse.ArgumentParser(description="Build MiniMind-ToolRL internship datasets.")
    parser.add_argument("--output_dir", default="../dataset", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--sft_size", default=5000, type=int)
    parser.add_argument("--dpo_size", default=2000, type=int)
    parser.add_argument("--rl_size", default=1000, type=int)
    parser.add_argument("--eval_size", default=200, type=int)
    args = parser.parse_args()

    dump_jsonl(os.path.join(args.output_dir, "intern_tool_sft.jsonl"), [sft_row(t) for t in build_rows(args.sft_size, args.seed)])
    rng = random.Random(args.seed + 1)
    dump_jsonl(os.path.join(args.output_dir, "intern_tool_dpo.jsonl"), [dpo_row(t, rng) for t in build_rows(args.dpo_size, args.seed + 2)])
    dump_jsonl(os.path.join(args.output_dir, "intern_tool_agent.jsonl"), [agent_row(t) for t in build_rows(args.rl_size, args.seed + 3)])
    dump_jsonl(os.path.join(args.output_dir, "intern_tool_eval.jsonl"), [eval_row(t) for t in build_rows(args.eval_size, args.seed + 4)])
    print(f"Wrote MiniMind-ToolRL datasets to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
