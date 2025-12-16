# /data/nvme0/yy/vt/MuMA-ToM/LIMP_Porker/muma_tom_pipeline/toM_latents.py
from typing import Dict

def build_latents_for_q2(state: dict) -> Dict[str, str]:
    """
    生成“谁当前胜率更高”的选项潜变量三元组
    参数:
      state: 统一扑克状态JSON
    返回:
      dict: { "A": "<Belief; Social goal; Believed Goal>", "B": "<...>" }
    """
    board = state.get("board", [])
    belief_sb = f"Board {','.join(board)} favors SB range"
    belief_bb = f"Board {','.join(board)} favors BB range"
    return {
        "A": f"Belief: {belief_sb}; Social goal: value; Believed Goal: BB wants to control pot",
        "B": f"Belief: {belief_bb}; Social goal: value; Believed Goal: SB wants to control pot",
    }

def build_latents_for_bluff(state: dict) -> Dict[str, str]:
    """
    生成“SB 当前行动更像：诈唬/价值”的选项潜变量三元组
    参数:
      state: 统一扑克状态JSON
    返回:
      dict: { "A": "<Belief; Social goal; Believed Goal>", "B": "<...>" }
    """
    return {
        "A": "Belief: SB has weak showdown; Social goal: bluff; Believed Goal: BB wants to fold",
        "B": "Belief: SB has strong made hand; Social goal: value; Believed Goal: BB wants to call",
    }