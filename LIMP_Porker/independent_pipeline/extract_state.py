import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROMPT = """
你将查看一组双人德州扑克的帧信息或文字描述，输出统一JSON结构：
{"blinds":{"SB":"P1","BB":"P2","sb_amount":5,"bb_amount":10},"players":[{"name":"P1","seat":1,"stack":300000},{"name":"P2","seat":2,"stack":240000}],"street":"flop","board":["Ah","7d","2c"],"hole_cards":{"P1":["Ks","Qs"],"P2":["??","??"]},"pot":44000,"actions":["P1 posts SB 5","P2 posts BB 10"]}
严格只输出JSON，不要额外文本。未知手牌用??。

补充说明：
这是一张双人德州扑克的某阶段的截图，左右两边为各自选手当前手头剩余筹码（Stack，单位为K，表示 "千美元"）、姓名、当前行动（行动里的金额单位为 "美元"），以及当前双方底牌（Hole Cards）和系统预测的胜率（带百分号）。中间上方红底白字表示当前底池（Pot）里的金额（单位 "美元"），"D" 标志所在的一侧为SB（Small blind，双人德州扑克里同时扮演庄家），另一方为BB（Big blind）。
"""

def extract_from_text(text: str) -> str:
    client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
    r = client.chat.completions.create(
        messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": text}],
        model=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
        temperature=0.0,
    )
    return r.choices[0].message.content.strip()

import base64

def _encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_from_images(image_paths):
    client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
    messages = [{"role": "system", "content": PROMPT}]
    for i, p in enumerate(image_paths):
        b64 = _encode_image(p)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Frame {i}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        })
    r = client.chat.completions.create(
        messages=messages,
        model=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
        temperature=0.0,
    )
    return r.choices[0].message.content.strip()

def parse_state(state_json: str):
    return json.loads(state_json)