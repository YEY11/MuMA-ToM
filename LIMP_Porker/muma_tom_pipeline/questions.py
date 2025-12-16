# python /data/nvme0/yy/vt/MuMA-ToM/LIMP_Porker/muma_tom_pipeline/questions.py
def _board_texture(board: list[str]) -> str:
    ranks_map = {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"T":10,"J":11,"Q":12,"K":13,"A":14}
    suits = [c[1] for c in board if isinstance(c, str) and len(c) >= 2]
    ranks = sorted([ranks_map.get(c[0].upper(), 0) for c in board if isinstance(c, str) and len(c) >= 2])
    max_suit = max([suits.count(s) for s in "cdhs"], default=0)
    flush_draw = max_suit >= 2
    conn = 0
    for i in range(1, len(ranks)):
        if ranks[i] - ranks[i - 1] <= 2:
            conn += 1
    wet = flush_draw or conn >= 2
    return "wet" if wet else "dry"


def _action_evidence(acts: list[str], pot: float) -> dict:
    text = " ".join(acts).lower()
    
    # Basic Style
    style = "aggressive" if any(k in text for k in ["bet", "raise", "all-in", "shove"]) else (
        "passive" if any(k in text for k in ["check", "call"]) else "unknown"
    )
    
    # Bet Size Ratio
    import re
    nums = [float(n) for n in re.findall(r"(\d+\.?\d*)", text)]
    amt = nums[-1] if nums else None
    ratio = None
    if amt is not None and pot and pot > 0:
        ratio = amt / float(pot)
    if ratio is None:
        size_cat = "unknown"
    elif ratio < 0.33:
        size_cat = "small"
    elif ratio < 0.66:
        size_cat = "medium"
    else:
        size_cat = "large"
        
    # Sequential Parsing (Mock for now, but structure ready for multi-step)
    # "P1 checks", "P2 bets 100", "P1 calls" -> [("check", 0), ("bet", 100), ("call", 0)]
    # This will be used by SequenceAgent
    parsed_sequence = []
    for a in acts:
        a_lower = a.lower()
        act_type = "unknown"
        if "check" in a_lower: act_type = "check"
        elif "call" in a_lower: act_type = "call"
        elif "fold" in a_lower: act_type = "fold"
        elif "raise" in a_lower: act_type = "raise"
        elif "bet" in a_lower: act_type = "bet"
        elif "all-in" in a_lower: act_type = "all-in"
        
        val = 0
        v_match = re.findall(r"(\d+\.?\d*)", a)
        if v_match:
            val = float(v_match[-1])
        parsed_sequence.append({"type": act_type, "amount": val, "raw": a})

    return {
        "style": style,
        "size_category": size_cat,
        "ratio": ratio,
        "sequence": parsed_sequence,
        "text": text
    }

def _rule_gold_q3(state: dict, equity: dict) -> str:
    acts = state.get("actions", [])
    pot = float(state.get("pot", 0) or 0)
    ev_dict = _action_evidence(acts, pot)
    style = ev_dict["style"]
    size = ev_dict["size_category"]
    texture = _board_texture(state.get("board", []))
    sb_equity = float(equity.get("SB_equity", 0.5))
    if style == "aggressive":
        if sb_equity < 0.5 and size in ("large", "medium") and texture == "dry":
            return "A"
        if sb_equity >= 0.5 and texture == "wet" and size in ("small", "medium"):
            return "B"
        return "B" if sb_equity >= 0.6 else "A"
    if style == "passive":
        return "B"
    return "B" if sb_equity >= 0.5 else "A"


def build_questions(state: dict, equity: dict, labels=None) -> dict:
    sb = state["blinds"]["SB"]
    bb = state["blinds"]["BB"]
    texture = _board_texture(state.get("board", []))
    ev_dict = _action_evidence(state.get("actions", []), float(state.get("pot", 0) or 0))
    
    sb_eq = float(equity.get("SB_equity", 0.5))
    bb_eq = float(equity.get("BB_equity", 0.5))
    
    # Comprehensive structured evidence for Agents
    evidence_struct = {
        "board_texture": texture,
        "action_analysis": ev_dict,
        "pot": state.get("pot", 0),
        "sb_equity": sb_eq,
        "bb_equity": bb_eq,
        "street": state.get("street", "unknown"),
        "position": "SB" # Assuming SB decision focus for Q3
    }
    
    # Legacy string evidence for simple prompt fallback
    evidence3_str = (
        f"Board={texture}, ActionStyle={ev_dict['style']}, BetSize={ev_dict['size_category']}, "
        f"Pot={state.get('pot', 0)}, SB_equity={sb_eq:.3f}, BB_equity={bb_eq:.3f}"
    )

    q = {
        "questions": {
            "1": f"谁是SB？A {sb}  B {bb}",
            "2": f"谁当前胜率更高？A {sb}  B {bb}",
            "3": f"SB 当前行动更像？A 诈唬  B 价值下注",
        },
        "options": {
            "1": ["A", "B"],
            "2": ["A", "B"],
            "3": ["A", "B"],
        },
        "mapping": {
            "1": {"A": sb, "B": bb},
            "2": {"A": sb, "B": bb},
            "3": {"A": "bluff", "B": "value"},
        },
        "answers": {
            "1": [labels.get("1")] if labels and labels.get("1") else ["A"],
            "2": [labels.get("2")] if labels and labels.get("2") else (["A"] if sb_eq > bb_eq else ["B"]),
            "3": [labels.get("3")] if labels and labels.get("3") else [_rule_gold_q3(state, equity)],
        },
        "evidence": {
            "3": evidence3_str,
            "structured": evidence_struct
        }
    }
    return q