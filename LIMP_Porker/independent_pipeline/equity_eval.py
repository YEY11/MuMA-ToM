import os
import random
import json
from dotenv import load_dotenv

load_dotenv()

def _import_eval7():
    try:
        import eval7
        return eval7
    except Exception as e:
        raise RuntimeError("eval7 is required for eval7 mode. Install with 'pip install eval7'.") from e

RANKS = "23456789TJQKA"
SUITS = "cdhs"

def _parse_card(s):
    eval7 = _import_eval7()
    return eval7.Card(s)

def _deck_without(exclude):
    eval7 = _import_eval7()
    deck = [eval7.Card(r + s) for r in RANKS for s in SUITS]
    excl = set(exclude)
    return [c for c in deck if c not in excl]

def eval7_equity(hole1, hole2, board=None, iters=10000):
    eval7 = _import_eval7()
    board = board or []
    h1_known = [c for c in hole1 if c != "??"]
    h2_known = [c for c in hole2 if c != "??"]
    known_cards = [
        *_to_cards(h1_known),
        *_to_cards(h2_known),
        *_to_cards(board),
    ]
    deck = _deck_without(known_cards)
    wins = ties = 0
    need_board = 5 - len(board)
    draw_needed = need_board + (2 - len(h1_known)) + (2 - len(h2_known))
    for _ in range(iters):
        draw = random.sample(deck, draw_needed)
        i = 0
        h1_fill = draw[i:i + (2 - len(h1_known))] if len(h1_known) < 2 else []
        i += (2 - len(h1_known)) if len(h1_known) < 2 else 0
        h2_fill = draw[i:i + (2 - len(h2_known))] if len(h2_known) < 2 else []
        i += (2 - len(h2_known)) if len(h2_known) < 2 else 0
        board_fill = draw[i:i + need_board]
        full_board = [*_to_cards(board), *board_fill]
        hand1_cards = [*_to_cards(h1_known), *h1_fill, *full_board]
        hand2_cards = [*_to_cards(h2_known), *h2_fill, *full_board]
        s1 = eval7.evaluate(hand1_cards)
        s2 = eval7.evaluate(hand2_cards)
        if s1 > s2:
            wins += 1
        elif s1 == s2:
            ties += 1
    p1 = (wins + 0.5 * ties) / iters
    return p1, 1.0 - p1

def _to_cards(cards):
    return [_parse_card(c) for c in cards]

from openai import OpenAI

def llm_equity(state_json):
    client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
    prompt = (
        "你将读取一段德州扑克当前状态的JSON，评估SB和BB的当前胜率。仅输出JSON，格式为 "
        "{\"SB_equity\": x.xx, \"BB_equity\": y.yy} ，数值为0-1之间。JSON如下：\n" + state_json
    )
    r = client.chat.completions.create(
        messages=[{"role": "system", "content": prompt}],
        model=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
        temperature=0.0,
    )
    content = r.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except Exception:
        return {"SB_equity": None, "BB_equity": None, "raw": content}