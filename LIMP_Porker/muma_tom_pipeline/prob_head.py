# python /data/nvme0/yy/vt/MuMA-ToM/LIMP_Porker/muma_tom_pipeline/prob_head.py
from toM_latents import build_latents_for_q2, build_latents_for_bluff
from openai import OpenAI
import os
import math
import json
from loguru import logger

client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))

# --- Multi-Agent Architecture Mocks ---
# In a full implementation, these would be separate modules or classes

class VisualAgent:
    def evaluate(self, evidence: dict, latents: dict) -> list[float]:
        # Analyze board texture, stack sizes, visual tells (if any)
        # For now, return neutral or bias slightly based on board texture
        logger.debug("[VisualAgent] Evaluating board texture and stacks")
        # Mock: if board is wet, slightly favor value (B) as it hits ranges more often? 
        # Just neutral for now to avoid noise
        return [0.5, 0.5]

class StrategyAgent:
    def evaluate(self, evidence: dict, latents: dict) -> list[float]:
        # Analyze bet sizing, equity, position
        logger.debug("[StrategyAgent] Evaluating bet size and equity")
        sb_eq = evidence.get("sb_equity", 0.5)
        # Simple heuristic bias: high equity -> value (B), low equity -> bluff (A)
        # This mimics a rational strategy assessment
        p_value = sb_eq
        return [1.0 - p_value, p_value]

class AudioAgent:
    def evaluate(self, evidence: dict, latents: dict) -> list[float]:
        # Analyze speech text sentiment/tone (if audio available)
        logger.debug("[AudioAgent] No audio evidence, returning neutral")
        return [0.5, 0.5]

class SequenceAgent:
    def evaluate(self, evidence: dict, latents: dict) -> list[float]:
        # Sequential Inversion:
        # P(Action_seq | Goal) = Product P(Action_t | Goal, History_t)
        logger.debug("[SequenceAgent] Performing sequential inversion")
        seq = evidence.get("action_analysis", {}).get("sequence", [])
        
        # Mock sequential scoring:
        # For each action, we'd ask LLM or use a model to score logprob(action | latent)
        # Here we simulate it by aggregating the "style" impact
        
        score_A = 0.0 # Log prob for Bluff
        score_B = 0.0 # Log prob for Value
        
        for act in seq:
            atype = act["type"]
            amt = act["amount"]
            
            # Mock conditional probs:
            if atype in ["bet", "raise", "all-in"]:
                # Aggressive action:
                # Bluffing (A): Consistent (high prob)
                # Value (B): Consistent (high prob)
                # Differentiate by sizing?
                # Large bet -> Polarized (Bluff or Nuts). 
                pass
            elif atype in ["check", "call"]:
                # Passive:
                # Bluff (A): Inconsistent (low prob) - bluffs usually bet
                # Value (B): Consistent (trapping/pot control)
                score_A -= 1.0
                score_B += 0.5
                
        # Convert log scores back to probs
        # Softmax([score_A, score_B])
        # Handle overflow
        m = max(score_A, score_B)
        ea = math.exp(score_A - m)
        eb = math.exp(score_B - m)
        s = ea + eb
        return [ea/s, eb/s]

# Instantiate Agents
visual_agent = VisualAgent()
strategy_agent = StrategyAgent()
audio_agent = AudioAgent()
sequence_agent = SequenceAgent()

def _fusion(probs_list: list[list[float]]) -> list[float]:
    # Product of Experts (PoE) or Weighted Average
    # Using PoE (multiplying probabilities) implies independent evidence sources
    
    final_A = 1.0
    final_B = 1.0
    
    for p in probs_list:
        final_A *= p[0]
        final_B *= p[1]
    
    s = final_A + final_B
    if s == 0: return [0.5, 0.5]
    return [final_A/s, final_B/s]


def _ab_likely(latent_A: str, latent_B: str, evidence_str: str) -> list[float]:
    # Legacy direct LLM call for consistency evaluation
    # Now treated as the "Narrative Consistency Agent"
    prompt = (
        "Evidence: {e}\n"
        "A: {a}\n"
        "B: {b}\n"
        "依据证据判断 A/B 哪个更可能。仅输出JSON：{{\"A_prob\": x, \"B_prob\": y}}，且 x+y=1。\n"
    ).format(e=evidence_str, a=latent_A, b=latent_B)
    logger.info("[ToM] Likelihood request")
    r = client.chat.completions.create(
        messages=[{"role":"system","content":prompt}],
        model=os.getenv("LLM_MODEL_NAME","gpt-4o"),
        temperature=0.0,
    )
    content = r.choices[0].message.content.strip()
    logger.debug(f"[ToM] Likelihood response={content[:200]}")
    try:
        obj = json.loads(content)
        pa = float(obj.get("A_prob", 0.5))
        pb = float(obj.get("B_prob", 0.5))
    except Exception:
        first = (content or "").strip()[:1]
        if first == "A":
            pa, pb = 1.0, 0.0
        elif first == "B":
            pa, pb = 0.0, 1.0
        else:
            pa, pb = 0.5, 0.5
    s = pa + pb
    pa = pa/s if s>0 else 0.5
    pb = pb/s if s>0 else 0.5
    return [pa, pb]

def probs_for_question(qid: str, question_pack: dict, state: dict, equity: dict) -> list[float]:
    logger.info(f"[Q{qid}] Compute probs")
    if qid == "1":
        return [1.0, 0.0]
    if qid == "2":
        sb_p = float(equity["SB_equity"])
        bb_p = float(equity["BB_equity"])
        lat = build_latents_for_q2(state)
        ev = f"SB_equity={sb_p:.3f}, BB_equity={bb_p:.3f}"
        # Q2 is primarily about equity, but we can still use ToM narrative check
        tom_probs = _ab_likely(lat["A"], lat["B"], ev)
        # 融合策略：胜率概率与 ToM 概率做乘性融合
        fused = [sb_p*tom_probs[0], bb_p*tom_probs[1]]
        # Normalize
        s = sum(fused)
        if s>0: fused = [x/s for x in fused]
        else: fused = [0.5, 0.5]
        logger.info(f"[Q2] fused probs: {fused}")
        return fused
        
    if qid == "3":
        lat = build_latents_for_bluff(state)
        
        # 1. Narrative Agent (LLM direct)
        ev_str = (question_pack.get("evidence", {}) or {}).get("3", "")
        p_narrative = _ab_likely(lat["A"], lat["B"], ev_str)
        logger.info(f"[Q3] Narrative Agent: {p_narrative}")
        
        # 2. Structured Evidence Agents
        struct_ev = (question_pack.get("evidence", {}) or {}).get("structured", {})
        
        p_visual = visual_agent.evaluate(struct_ev, lat)
        p_strategy = strategy_agent.evaluate(struct_ev, lat)
        p_audio = audio_agent.evaluate(struct_ev, lat)
        p_sequence = sequence_agent.evaluate(struct_ev, lat)
        
        logger.info(f"[Q3] Sub-Agents: Vis={p_visual}, Strat={p_strategy}, Seq={p_sequence}")
        
        # 3. Fusion
        # We can weight them. For now, equal weight PoE
        # Narrative agent is usually strong, so we include it
        fused = _fusion([p_narrative, p_visual, p_strategy, p_audio, p_sequence])
        
        logger.info(f"[Q3] Final Fused probs: {fused}")
        return fused
        
    return [0.5, 0.5]