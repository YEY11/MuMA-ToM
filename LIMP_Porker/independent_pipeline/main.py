import os
import json
from dotenv import load_dotenv
from equity_eval import eval7_equity, llm_equity
from extract_state import extract_from_text, extract_from_images, parse_state

load_dotenv()

def load_episode_text(ep_dir):
    text_path = os.path.join(ep_dir, "desc.txt")
    if os.path.isfile(text_path):
        with open(text_path, "r") as f:
            return f.read()
    return None

def choose_equity_mode():
    mode = os.getenv("POKER_EQUITY_MODE", "eval7")
    return mode

def equity_from_state(state):
    mode = choose_equity_mode()
    if mode == "llm":
        return llm_equity(json.dumps(state, ensure_ascii=False))
    sb_name = state["blinds"]["SB"]
    bb_name = state["blinds"]["BB"]
    hole = state.get("hole_cards", {})
    board = state.get("board", [])
    h_sb = hole.get(sb_name, ["??", "??"])
    h_bb = hole.get(bb_name, ["??", "??"])
    p_sb, p_bb = eval7_equity(h_sb, h_bb, board)
    return {"SB_equity": p_sb, "BB_equity": p_bb}

def run_episode(ep_dir):
    text = load_episode_text(ep_dir)
    state_json = None
    if text:
        state_json = extract_from_text(text)
    else:
        imgs = [os.path.join(ep_dir, f) for f in os.listdir(ep_dir) if f.lower().endswith((".png",".jpg",".jpeg"))]
        if not imgs:
            raise FileNotFoundError(f"No desc.txt and no images in {ep_dir}")
        state_json = extract_from_images(sorted(imgs)[:10])
    state = parse_state(state_json)
    eq = equity_from_state(state)
    print(json.dumps({"state": state, "equity": eq}, ensure_ascii=False))

if __name__ == "__main__":
    root = os.getenv("POKER_DATASET_ROOT", "/data/nvme0/yy/vt/MuMA-ToM/datasets/porker/")
    episode = os.getenv("POKER_EPISODE", "episode_1")
    ep_dir = os.path.join(root, episode)
    run_episode(ep_dir)