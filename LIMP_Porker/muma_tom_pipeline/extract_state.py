# python /data/nvme0/yy/vt/MuMA-ToM/LIMP_Porker/muma_tom_pipeline/extract_state.py
import os, json, base64
from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger

load_dotenv()

PROMPT = """
你将查看一组双人德州扑克的帧信息或文字描述，输出统一JSON结构：
{"blinds":{"SB":"P1","BB":"P2","sb_amount":5,"bb_amount":10},"players":[{"name":"P1","seat":1,"stack":300000},{"name":"P2","seat":2,"stack":240000}],"street":"flop","board":["Ah","7d","2c"],"hole_cards":{"P1":["Ks","Qs"],"P2":["??","??"]},"pot":44000,"actions":["P1 posts SB 5","P2 posts BB 10"]}
严格只输出JSON，不要额外文本。未知手牌用??。
补充说明：
这是一张双人德州扑克的某阶段的截图，左右两边为各自选手当前手头剩余筹码（Stack，单位为K，表示 "千美元"）、姓名、当前行动（行动里的金额单位为 "美元"），以及当前双方底牌（Hole Cards）和系统预测的胜率（带百分号）。中间上方红底白字表示当前底池（Pot）里的金额（单位 "美元"），"D" 标志所在的一方为SB（Small blind，双人德州扑克里同时扮演庄家），另一方为BB（Big blind）。
"""

def _client():
    return OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))

def extract_from_text(text: str) -> str:
    logger.info("[STATE] From text")
    r = _client().chat.completions.create(
        messages=[{"role":"system","content":PROMPT},{"role":"user","content":text}],
        model=os.getenv("LLM_MODEL_NAME","gpt-4o"),
        temperature=0.0,
    )
    content = r.choices[0].message.content.strip()
    logger.debug(f"[STATE] Text response={content[:200]}")
    return content

def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return "image/jpeg" if ext in [".jpg",".jpeg"] else "image/png"

def extract_from_images(image_paths: list[str]) -> str:
    logger.info(f"[STATE] From images frames={len(image_paths)}")
    messages = [{"role":"system","content":PROMPT}]
    for i,p in enumerate(image_paths):
        b64 = _encode_image(p)
        messages.append({
            "role":"user",
            "content":[
                {"type":"text","text":f"Frame {i}"},
                {"type":"image_url","image_url":{"url":f"data:{_mime(p)};base64,{b64}"}}
            ]
        })
    r = _client().chat.completions.create(
        messages=messages,
        model=os.getenv("LLM_MODEL_NAME","gpt-4o"),
        temperature=0.0,
    )
    content = r.choices[0].message.content.strip()
    logger.debug(f"[STATE] Images response={content[:200]}")
    return content

def parse_state(state_json: str) -> dict:
    logger.info("[STATE] Parse json")
    obj = json.loads(state_json)
    logger.debug(f"[STATE] Keys={list(obj.keys())}")
    return obj