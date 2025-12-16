import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger
from typing import Dict, Any, List, Optional

load_dotenv()

class ToMReasoningAgent:
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
        self.model = model

    def predict_winner(self, unified_state: Dict[str, Any], current_phase: str, multimodal: bool = True) -> Dict[str, Any]:
        """
        Predict the winner of the hand based on available information up to the current phase.
        
        Args:
            unified_state: The full perception JSON.
            current_phase: "Pre-flop", "Flop", "Turn", or "River".
            multimodal: If True, includes audio insights and facial emotions.
        """
        context = self._build_context(unified_state, current_phase, multimodal)
        
        prompt = f"""
        You are a high-stakes poker analyst and psychologist.
        Your task is to predict the WINNER of this hand based on the provided context.
        
        Context:
        {context}
        
        Task:
        1. Analyze the mathematical strength of each player's hand (Game Theory).
        2. {"Analyze the psychological state and potential bluffs based on emotions and commentary (Theory of Mind)." if multimodal else "Ignore psychological cues, focus only on game state."}
        3. Predict who will win the pot.
        
        Output JSON:
        {{
            "predicted_winner": "Player Name",
            "confidence": 0.0 to 1.0,
            "reasoning": "Step-by-step analysis...",
            "key_factors": ["List of decisive factors"]
        }}
        """
        
        return self._call_llm(prompt)

    def detect_bluff(self, unified_state: Dict[str, Any], current_phase: str, player_name: str, multimodal: bool = True) -> Dict[str, Any]:
        """
        Assess if a specific player is bluffing.
        """
        context = self._build_context(unified_state, current_phase, multimodal)
        
        prompt = f"""
        You are a poker behavior expert.
        Your task is to determine if {player_name} is BLUFFING in the {current_phase} phase.
        
        Context:
        {context}
        
        Task:
        1. Evaluate {player_name}'s hand strength vs the board.
        2. {"Analyze facial expressions (micro-expressions, eye contact) and commentary insights." if multimodal else "Focus only on betting patterns and hand strength."}
        3. Decide if their actions represent a Bluff or Value.
        
        Output JSON:
        {{
            "is_bluffing": true/false,
            "probability": 0.0 to 1.0,
            "reasoning": "Analysis...",
            "evidence": ["List of supporting evidence"]
        }}
        """
        
        return self._call_llm(prompt)

    def _build_context(self, state: Dict[str, Any], target_phase: str, multimodal: bool) -> str:
        # 1. Game State Construction (Cumulative up to target_phase)
        game_log = []
        target_phase_data = None
        
        # Timeline is a list of phases
        timeline = state.get("timeline", [])
        
        # Audio insights are global usually, but we can try to filter if needed. 
        # For now, we provide global insights as "General Context".
        audio_insights = state.get("commentary_insights", {})
        
        found_phase = False
        for phase_data in timeline:
            p_name = phase_data["phase"]
            
            # Game State info
            gs = phase_data.get("game_state", {})
            players = gs.get("players", [])
            board = gs.get("board", [])
            pot = gs.get("pot", 0)
            
            phase_info = f"--- Phase: {p_name} ---\n"
            phase_info += f"Pot: {pot}\nBoard: {board}\n"
            for p in players:
                phase_info += f"Player {p.get('name')} (Stack: {p.get('stack')})\n"
                # Hole cards are only visible in some datasets, assume visible for "God Mode" reasoning
                # If we want to simulate player perspective, we'd hide opponent cards.
                # For this task, let's assume God Mode (Spectator) first.
                hole = gs.get("hole_cards", {}).get(f"P{p.get('seat')}", []) # Mapping seat to P1/P2 needs care
                # Try to match seat/name to hole cards if possible, or just dump hole_cards
                # Simple dump:
            phase_info += f"Hole Cards (All): {json.dumps(gs.get('hole_cards', {}))}\n"
            
            # Multimodal: Emotions
            if multimodal:
                emotions = phase_data.get("scene_emotions", [])
                phase_info += f"Facial Emotions: {json.dumps(emotions)}\n"
            
            game_log.append(phase_info)
            
            if p_name == target_phase:
                found_phase = True
                break
        
        context_str = "\n".join(game_log)
        
        if multimodal:
            context_str += "\n\n--- Commentary & Audio Insights ---\n"
            context_str += json.dumps(audio_insights, indent=2)
            
        return context_str

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return {"error": str(e)}