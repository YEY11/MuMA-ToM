# python /data/nvme0/yy/vt/MuMA-ToM/LIMP_Porker/muma_tom_pipeline/equity_eval.py
import os, random, json
from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger

load_dotenv()
RANKS = "23456789TJQKA"
SUITS = "cdhs"

def _import_eval7():
    import eval7
    return eval7

def _normalize_card(s: str):
    if not s:
        return None
    s = s.strip()
    if s == "??":
        return None
    if s.startswith("10"):
        r = "T"
        su = s[-1]
    else:
        r = s[0].upper()
        su = s[-1]
    suit_map = {"♣":"c","♦":"d","♥":"h","♠":"s","C":"c","D":"d","H":"h","S":"s"}
    su = suit_map.get(su, su.lower())
    if r in RANKS and su in SUITS:
        return r + su
    return None

def _parse_card(s):
    return _import_eval7().Card(s)

def _deck_without(exclude_cards):
    ev = _import_eval7()
    deck = [ev.Card(r+s) for r in RANKS for s in SUITS]
    excl = set(exclude_cards)
    return [c for c in deck if c not in excl]

def _to_cards(cards):
    return [_parse_card(c) for c in cards]

def eval7_equity(hole1, hole2, board=None, iters=10000):
    logger.info("[EQUITY] eval7 start")
    ev = _import_eval7()
    board = board or []
    h1_known = [c for c in map(_normalize_card, hole1) if c is not None]
    h2_known = [c for c in map(_normalize_card, hole2) if c is not None]
    board_known = [c for c in map(_normalize_card, board) if c is not None]
    logger.debug(f"[EQUITY] eval7 inputs h1={h1_known}, h2={h2_known}, board={board_known}")
    known = [*_to_cards(h1_known), *_to_cards(h2_known), *_to_cards(board_known)]
    deck = _deck_without(known)
    wins = ties = 0
    need_board = 5 - len(board_known)
    draw_needed = need_board + (2 - len(h1_known)) + (2 - len(h2_known))
    for _ in range(iters):
        draw = random.sample(deck, draw_needed)
        i = 0
        h1_fill = draw[i:i+(2-len(h1_known))] if len(h1_known) < 2 else []
        i += (2-len(h1_known)) if len(h1_known) < 2 else 0
        h2_fill = draw[i:i+(2-len(h2_known))] if len(h2_known) < 2 else []
        i += (2-len(h2_known)) if len(h2_known) < 2 else 0
        board_fill = draw[i:i+need_board]
        full_board = [*_to_cards(board_known), *board_fill]
        s1 = ev.evaluate([*_to_cards(h1_known), *h1_fill, *full_board])
        s2 = ev.evaluate([*_to_cards(h2_known), *h2_fill, *full_board])
        if s1 > s2: wins += 1
        elif s1 == s2: ties += 1
    p1 = (wins + 0.5*ties) / iters
    logger.info(f"[EQUITY] eval7 result SB={p1:.4f}, BB={1.0-p1:.4f}")
    return p1, 1.0 - p1

def llm_equity(state_json: str):
    logger.info("[EQUITY] llm start")
    client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
    prompt = "仅输出JSON：{\"SB_equity\": x.xx, \"BB_equity\": y.yy}。状态如下：\n" + state_json
    r = client.chat.completions.create(
        messages=[{"role":"system","content":prompt}],
        model=os.getenv("LLM_MODEL_NAME","gpt-4o"),
        temperature=0.0,
    )
    content = r.choices[0].message.content.strip()
    logger.debug(f"[EQUITY] llm response={content[:200]}")
    try:
        obj = json.loads(content)
        logger.info(f"[EQUITY] llm parsed SB={obj.get('SB_equity')}, BB={obj.get('BB_equity')}")
        return obj
    except Exception:
        logger.warning("[EQUITY] llm json parse failed")
        return {"SB_equity": None, "BB_equity": None, "raw": content}