import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Setup
load_dotenv()
api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
client = OpenAI(api_key=api_key, base_url=base_url)

def build_context(state, multimodal):
    log = []
    # Add Audio Insights
    if multimodal:
        insights = state.get("commentary_insights", {})
        log.append(f"--- Commentary Insights ---\n{json.dumps(insights, indent=2)}")
    
    # Add Timeline
    for phase in state.get("timeline", []):
        p_name = phase["phase"]
        gs = phase["game_state"]
        
        # Basic Game State
        info = f"\nPhase: {p_name}\nBoard: {gs.get('board')}\nPot: {gs.get('pot')}"
        info += f"\nPlayers: {json.dumps(gs.get('players'))}"
        # HIDE HOLE CARDS to test true ToM reasoning (Imperfect Information)
        # info += f"\nHole Cards: {json.dumps(gs.get('hole_cards'))}"
        
        # Multimodal: Emotions
        if multimodal:
            emotions = phase.get("scene_emotions", [])
            info += f"\nEmotions: {json.dumps(emotions)}"
            
        log.append(info)
        
    return "\n".join(log)

def query_llm(context, multimodal):
    mode = "MULTIMODAL" if multimodal else "BASELINE (Game State Only)"
    prompt = f"""
    You are a Poker Analyst.
    Mode: {mode}
    
    Context:
    {context}
    
    Task:
    Predict the winner of this hand and explain why.
    If Mode is MULTIMODAL, you MUST use the commentary and emotions to refine your prediction (e.g. is someone bluffing? is someone nervous?).
    If Mode is BASELINE, ignore all psychological/emotional cues.
    
    Output JSON:
    {{
        "winner": "Name",
        "reasoning": "...",
        "confidence": 0.0-1.0
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

# Load Data
with open("/Users/andy/Projects/MuMA-ToM/datasets/poker/episode_example/game1/unified_perception.json") as f:
    data = json.load(f)

# Run Comparison
print("Running Baseline (Game State Only)...")
ctx_base = build_context(data, multimodal=False)
res_base = query_llm(ctx_base, multimodal=False)

print("\nRunning Multimodal (Game State + Audio + Face)...")
ctx_multi = build_context(data, multimodal=True)
res_multi = query_llm(ctx_multi, multimodal=True)

print("\n" + "="*50)
print("COMPARISON RESULTS")
print("="*50)
print(f"BASELINE PREDICTION:\nWinner: {res_base.get('winner')}\nConf: {res_base.get('confidence')}\nReason: {res_base.get('reasoning')}\n")
print("-" * 50)
print(f"MULTIMODAL PREDICTION:\nWinner: {res_multi.get('winner')}\nConf: {res_multi.get('confidence')}\nReason: {res_multi.get('reasoning')}\n")