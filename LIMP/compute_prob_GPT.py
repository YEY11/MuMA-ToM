# LIMP 概率评估模块：基于潜变量评估话语/动作的一致性概率
# 统一使用 .env 的 LLM 配置
# LIMP 概率评估模块：基于潜变量评估话语/动作的一致性概率
# 统一使用 .env 的 LLM 配置
from openai import OpenAI
import json
import math
import re
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

def parse_latent_var(latent_var: str) -> Dict[str, str]:
    """解析潜变量字符串为字典：Belief/Social Goal/Believed Goal
    参数:
        latent_var (str): 潜变量字符串
    返回:
        Dict[str, str]: 包含 Belief/Social Goal/Believed Goal 的字典
    """
    belief = re.search(r'Belief:\s*(.*?)(?=\; Social goal)', latent_var).group(1)
    social_goal = re.search(r'Social goal:\s*(.*?)(?=\; Believed Goal)', latent_var).group(1)
    believed_goal = re.search(r'Believed Goal:\s*(.*)', latent_var).group(1)
    return {"Belief": belief, "Social Goal": social_goal, "Believed Goal": believed_goal}

def compute_prob(init_state: str, latent_var: str, info: Dict[str, Dict[str, Optional[List[str]]]], main_person: str, prompt: str) -> float:
    """整合话语与动作的可能性，连乘得到选项概率
    参数:
        init_state (str): 环境初始状态
        latent_var (str): 潜变量字符串
        info (Dict[str, Dict[str, Optional[List[str]]]]): 包含被评估者与评估者的话语与动作信息
        main_person (str): 当前主评估者姓名
        prompt (str): 评估提示模板
    返回:
        float: 选项总体可能性
    """
    latent_vars = parse_latent_var(latent_var)
    belief = latent_vars["Belief"]
    social_goal = latent_vars["Social Goal"]
    believed_goal = latent_vars["Believed Goal"]
    names = list(info.keys())
    other_name = [name for name in names if not name == main_person][0]
    if info[main_person]["utterance"] is not None:
        probability = compute_prob_utterance(other_name, main_person, info[other_name]["utterance"][0], info[main_person]["utterance"][0], social_goal, belief, believed_goal, None, exclude=["Believed_Goal"])
    else:
        probability = 1.0
    if info[main_person]["action"] is not None:
        for index, action in enumerate(info[main_person]["action"]):
            previous_actions = f"{other_name}'s actions:\n"
            for action1 in info[other_name]["action"]:
                previous_actions += action1
                previous_actions += "\n"
            previous_actions += f"{main_person}'s actions:\n"
            for i in range(index):
                previous_actions += info[main_person]["action"][i]
                previous_actions += "\n"
            prob = compute_prob_action(other_name, main_person, init_state, previous_actions, action, social_goal, belief, believed_goal)
            print(f"Probability of step {index}: {prob}")
            probability = probability * prob
    return probability


def compute_prob_utterance(name_agent_0: str, name_agent_1: str, utterance_agent_0: str, utterance_agent_1: str, a1_social_goal: str, a1_belief: str, a1_belief_of_goal: str, init_state: Optional[str], exclude: List[str] = []) -> float:
    """在给定潜变量下评估话语是否"可能"，返回 A(可能) 的概率
    参数:
        name_agent_0 (str): 被评估者姓名
        name_agent_1 (str): 评估者姓名
        utterance_agent_0 (str): 被评估者话语
        utterance_agent_1 (str): 评估者话语
        a1_social_goal (str): 当前评估者社会目标
        a1_belief (str): 当前评估者信念
        a1_belief_of_goal (str): 当前评估者信念关于被评估者目标
        init_state (Optional[str]): 环境初始状态
        exclude (List[str]): 排除项，默认不排除 "Believed_Goal"
    返回:
        Optional[float]: A(可能) 的概率，若无法计算则返回 None
    """
    evaluation_prompt = f"""
    {name_agent_1}'s social goal: {a1_social_goal}
    {name_agent_1}'s belief: {a1_belief}
    """
    if "Believed_Goal" not in exclude:
        evaluation_prompt += f"{name_agent_1}'s belief of {name_agent_0}'s goal: {a1_belief_of_goal}\n"
    evaluation_prompt += f"{name_agent_0}'s Utterance': {utterance_agent_0}\n"
    if init_state is not None:
        evaluation_prompt += f"Initial state of environment: {init_state}\n"
    evaluation_prompt += f"""
    Based on the information, decide if it is likely for {name_agent_1} to say this word given conditions above. Compare the utterance and the belief of {name_agent_1}. 
    When trying to hinder, {name_agent_1} is likely to give different information with belief. For example, saying that some object is there when {name_agent_1} believe that there is some other things or nothing there, or the object is at a different place.
    Respond with only either A or B:
    {name_agent_1}'s Utterance: {utterance_agent_1}
    A) Likely
    B) Unlikely
    """
    response2 = client.chat.completions.create(
        messages=[
            {"role": "system", "content": evaluation_prompt},
        ],
        model=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
        logprobs=True,
        top_logprobs=5,
        temperature=0.0
    )

    response_json_str = response2.model_dump_json(indent=2)
    response_dict = json.loads(response_json_str)
    logprob_a = None

    for top_logprob in response_dict['choices'][0]['logprobs']['content'][0]['top_logprobs']:
        if top_logprob['token'] == 'A':
            logprob_a = top_logprob['logprob']
        elif top_logprob['token'] == 'B':
            logprob_b = top_logprob['logprob']

    prob_a = math.exp(logprob_a) if logprob_a is not None else 0.0
    return prob_a


def compute_prob_action(name_agent_0: str, name_agent_1: str, init_state: str, previous_actions: str, a1_action: str, a1_social_goal: str, a1_belief: str, a1_belief_of_goal: str) -> Optional[float]:
    """
    在给定潜变量与历史动作下评估当前动作的"可能"性，返回 A(可能) 的概率
    参数:
        name_agent_0 (str): 被评估者姓名
        name_agent_1 (str): 评估者姓名
        init_state (str): 环境初始状态
        previous_actions (str): 历史动作序列
        a1_action (str): 当前评估者动作
        a1_social_goal (str): 当前评估者社会目标
        a1_belief (str): 当前评估者信念
        a1_belief_of_goal (str): 当前评估者信念关于被评估者目标
    返回:
        Optional[float]: A(可能) 的概率，若无法计算则返回 None
    """
    
    evaluation_prompt = f"""
    Decide if {name_agent_1}'s action is likely with the information provided, respond with only either A or B:
    {name_agent_0}'s social goal: {a1_social_goal}
    {name_agent_1}'s belief: {a1_belief}
    {name_agent_1}'s belief of {name_agent_0}'s goal: {a1_belief_of_goal}
    Initial state: {init_state}
    Check {name_agent_0}'s action to get the location of object when {name_agent_1} starts to act. 
    When {name_agent_1} tries to hinder, it's likely to grab object from its believed goal location for other agent, and unlikely to move objects to the believed goal location
    When {name_agent_1} tries to help, it's likely to grab object from somewhere else and put it to believed goal location, and unlikely to grab object from believed goal location
    Walking towards or grabbing from some unrelated location should be considered likely
    Previous Actions: {previous_actions}
    {name_agent_1}'s Action: {a1_action}
    A) Likely
    B) Unlikely
    """

    response2 = client.chat.completions.create(
        messages=[
            {"role": "system", "content": evaluation_prompt},
        ],
        model=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
        logprobs=True,
        top_logprobs=5,
        temperature=0.0
    )

    response_json_str = response2.model_dump_json(indent=2)
    response_dict = json.loads(response_json_str)
    logprob_a = None

    for top_logprob in response_dict['choices'][0]['logprobs']['content'][0]['top_logprobs']:
        if top_logprob['token'] == 'A':
            logprob_a = top_logprob['logprob']
        elif top_logprob['token'] == 'B':
            logprob_b = top_logprob['logprob']

    prob_a = math.exp(logprob_a) if logprob_a is not None else None
    return prob_a
