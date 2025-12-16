# python /data/nvme0/yy/vt/MuMA-ToM/LIMP_Porker/muma_tom_pipeline/main.py
import os, json
from dotenv import load_dotenv
from loguru import logger
from extract_state import extract_from_text, extract_from_images, parse_state
from equity_eval import eval7_equity, llm_equity
from questions import build_questions
from prob_head import probs_for_question

load_dotenv()

def _load_text(ep_dir):
    p = os.path.join(ep_dir, "desc.txt")
    if os.path.isfile(p):
        return open(p,"r").read()
    # Fallback to desc_copy.txt if exists (for testing)
    p2 = os.path.join(ep_dir, "desc_copy.txt")
    if os.path.isfile(p2):
        return open(p2,"r").read()
    return None

def _extract_state(ep_dir):
    logger.info(f"_extract_state: dir={ep_dir}")
    text = _load_text(ep_dir)
    if text:
        state = parse_state(extract_from_text(text))
        logger.debug(f"_extract_state: from text")
        return state
    imgs = [os.path.join(ep_dir,f) for f in os.listdir(ep_dir) if f.lower().endswith((".png",".jpg",".jpeg"))]
    if not imgs:
        raise FileNotFoundError(f"No desc.txt or images in {ep_dir}")
    state = parse_state(extract_from_images(sorted(imgs)[:10]))
    logger.debug(f"_extract_state: from images={len(imgs)}")
    return state

def _equity(state):
    mode = os.getenv("POKER_EQUITY_MODE","eval7")
    sb = state["blinds"]["SB"]
    bb = state["blinds"]["BB"]
    h = state.get("hole_cards",{})
    b = state.get("board",[])
    logger.info(f"_equity: mode={mode}")
    if mode == "llm":
        eq = llm_equity(json.dumps(state, ensure_ascii=False))
        logger.info(f"_equity: llm={eq}")
        return eq
    p_sb, p_bb = eval7_equity(h.get(sb,["??","??"]), h.get(bb,["??","??"]), b)
    eq = {"SB_equity": p_sb, "BB_equity": p_bb}
    logger.info(f"_equity: eval7={eq}")
    return eq

def _argmax(probs):
    return ["A","B","C","D","E"][max(range(len(probs)), key=lambda i: probs[i])]

def _load_labels(ep_dir):
    p = os.path.join(ep_dir, "labels.json")
    if os.path.isfile(p):
        try:
            return json.load(open(p, "r"))
        except Exception:
            return None
    return None

def run_episode(ep_dir):
    logger.info("[START] Episode pipeline")
    state = _extract_state(ep_dir)
    logger.info("[STATE] Extracted")
    equity = _equity(state)
    logger.info("[EQUITY] Computed")
    labels = _load_labels(ep_dir)
    logger.info(f"[LABELS] Loaded={'yes' if labels else 'no'}")
    qp = build_questions(state, equity, labels)
    logger.info("[QUESTIONS] Built")
    correct = 0; total = 0
    for qid, qtext in qp["questions"].items():
        logger.info(f"[Q{qid}] Begin")
        probs = probs_for_question(qid, qp, state, equity)
        choice = _argmax(probs)
        gold = qp["answers"][qid][0]
        logger.info(f"[Q{qid}] choice={choice} gold={gold} probs={probs}")
        print(json.dumps({"qid": qid, "question": qtext, "probs": probs, "choice": choice, "gold": gold}, ensure_ascii=False))
        if choice == gold: correct += 1
        total += 1
        logger.info(f"[Q{qid}] End")
    acc = correct/total if total>0 else 0.0
    logger.info(f"[END] accuracy={acc:.3f}")
    print(json.dumps({"accuracy": acc, "state": state, "equity": equity}, ensure_ascii=False))

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger.add(
        os.path.join(logs_dir, "muma_tom_pipeline.log"),
        rotation="1 week",
        retention="30 days",
        enqueue=True,
        level="INFO",
        format="{time} | {level} | {module}:{function}:{line} | {extra[rid]} | {message}"
    )
    root = os.getenv("POKER_DATASET_ROOT","/data/nvme0/yy/vt/MuMA-ToM/datasets/poker/")
    episode = os.getenv("POKER_EPISODE","episode_1")
    with logger.contextualize(rid=episode):
        run_episode(os.path.join(root, episode))
