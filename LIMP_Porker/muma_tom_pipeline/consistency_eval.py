import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger
from typing import Dict, Any, List

load_dotenv()

class ConsistencyEvaluator:
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
        self.model = model

    def evaluate_consistency(self, latents: Dict[str, str], evidence: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate how consistent each latent hypothesis (A/B) is with the observed evidence.
        
        Args:
            latents: {"A": "Belief: ...", "B": "Belief: ..."}
            evidence: {
                "game_state": {...},
                "audio_insights": {...},
                "scene_emotions": [...]
            }
            
        Returns:
            {"A": 0.8, "B": 0.2}
        """
        evidence_str = self._format_evidence(evidence)
        
        prompt = f"""
        You are a Theory of Mind (ToM) Reasoning Engine for Poker.
        Your goal is to determine which internal mental state (Latent A or B) best explains the observed behavior.
        
        OBSERVED EVIDENCE:
        {evidence_str}
        
        HYPOTHESES:
        Option A: {latents.get('A')}
        Option B: {latents.get('B')}
        
        TASK:
        1. Analyze the compatibility of the Evidence with Hypothesis A.
        2. Analyze the compatibility of the Evidence with Hypothesis B.
        3. Pay special attention to "Inconsistency":
           - If Audio says "He is steamy" but Hypothesis says "Rational Value Bet", that's inconsistent.
           - If Face says "Nervous" but Hypothesis says "Strong Hand", that's inconsistent (or implies bluffing).
           
        OUTPUT (JSON):
        {{
            "A_score": <0-100>,
            "B_score": <0-100>,
            "reasoning": "Step-by-step analysis citing specific evidence..."
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            result = json.loads(response.choices[0].message.content)
            
            # Normalize to probability
            score_a = result.get("A_score", 50)
            score_b = result.get("B_score", 50)
            total = score_a + score_b + 1e-6
            
            return {
                "A": score_a / total,
                "B": score_b / total,
                "reasoning": result.get("reasoning", "")
            }
            
        except Exception as e:
            logger.error(f"Consistency evaluation failed: {e}")
            return {"A": 0.5, "B": 0.5, "error": str(e)}

    def _format_evidence(self, evidence: Dict[str, Any]) -> str:
        lines = []
        
        # 1. Game Situation
        gs = evidence.get("game_state", {})
        lines.append(f"--- Game State ---")
        lines.append(f"Board: {gs.get('board')}")
        lines.append(f"Pot: {gs.get('pot')}")
        
        # 2. Audio / Commentary
        lines.append(f"\n--- Commentary Insights ---")
        insights = evidence.get("audio_insights", {})
        if insights:
            lines.append(json.dumps(insights, indent=2))
        else:
            lines.append("No commentary available.")
            
        # 3. Facial Emotions
        lines.append(f"\n--- Facial Expressions ---")
        emotions = evidence.get("scene_emotions", [])
        if emotions:
            lines.append(json.dumps(emotions, indent=2))
        else:
            lines.append("No facial data available.")
            
        return "\n".join(lines)