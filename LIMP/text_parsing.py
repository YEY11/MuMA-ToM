from openai import OpenAI
import os
import json
import re
import ipdb
import ast
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()
# LIMP 文本解析模块：抽取人物的动作与话语，生成 ToM 潜变量候选
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

latent_variable_prompt = """
    You will read a question about agents' mind and ideas, and the initial state of the environment from which agents' are interacting in. Agents' knowledge & belief are about this initial state, but not necessarily changed state after some actions. For each choice, extract one set of second person's belief (make sure to turn it into some statement about the environment state), second person's social goal toward first peron's actions (help, hinder or some similar words of indepedent), and second person's believed first person's physical goal (some arrangement of objects). Organize the answer in this way: A: Belief: contents; Social goal: contents; Believed Goal: contents. B: Belief: contents; Social goal: contents; Believed Goal: contents. C: Belief: contents; Social goal: contents; Believed Goal: contents. Do not include any other information or extra contents. Make sure your answer follow the format requirement, use ";" to separate variables within each choice and end response with ".". Separate contents of "A", "B" and "C" with "."

    Question: {}
"""

# 从动作序列反推出“初始环境状态”的提示：仅提取环境状态，不包含其他信息
init_state_prompt = """
    You will read one or two person's actions in a list like form. From the actions taken, extract the initial state of the environment before any people act. 
    Check each grab action or synonyms. Describe it in the form "There is a [object grabbed] [on/inside location of grabbing].
    Only include environment states statements. Do not include any other information or extra contents.

    Actions: {}
"""


def parse_text_info(text: str, name: str) -> Dict[str, Optional[List[str]]]:
    """解析指定人物在文本中的 Actions/Utterance，返回列表字典
    参数:
        text (str): 叙述文本
        name (str): 目标人物姓名
    返回:
        dict: {"action": list[str] | None, "utterance": list[str] | None}
    """
    info_extraction_prompt = """
        You will read a piece of text describing actions of some number of people with distinctive names. You will also have a name, which is the name of the person whom you should pay attention to. Summarize the person's actions and utterance separately in a chronological order. Only include the actions and utterance directly taken by the person in the text, and exclude any previous actions mentioned indirectly. If you cannot find either utterance or actions of the person in the text, leave the corresponding section blank. When reading words like "it", replace it with inferred object or location to make actions clearer. Do not include agent's communication as part of it. Organize your answer in this form:
        Actions:
        ["action one", "action two", "action three", ...]
        ...
        Utterance:
        ["utterance one", "utterance two", "utterance three", ...]
        ... 

        Text: {}

        Name: {}
        
    """
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": info_extraction_prompt.format(text, name)},
        ],
        model=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
        temperature=0.0
    )
    info = response.choices[0].message.content.strip()
    actions_match = re.search(r'Actions:\s*(\[[^\]]*\])', info)
    utterance_match = re.search(r'Utterance:\s*(\[[^\]]*\])', info)

    actions = actions_match.group(1) if actions_match else None
    utterance = utterance_match.group(1) if utterance_match else None

    action_list = ast.literal_eval(actions)
    utterance_list = ast.literal_eval(utterance)
    if len(action_list) == 0:
        action_list = None
    if len(utterance_list) == 0:
        utterance_list = None
    return {"action": action_list, "utterance": utterance_list}

def latent_variable_extraction(info: Dict[str, Dict[str, Optional[List[str]]]], question: str) -> Tuple[str, Dict[str, Optional[str]]]:
    """根据解析出的动作，先抽取初始环境状态，再为每个选项生成 ToM 潜变量三元组
    参数:
        info (dict): name -> {action, utterance}
        question (str): 问题文本
    返回:
        Tuple[str, Dict[str, Optional[str]]]: (init_state(str), choices(dict))，choices 包含 A/B/C 的三元组字符串
    """
    action_str = ""
    print(info)
    for name in info.keys():
        if info[name]["action"] is not None:
            action_str += f"{name}'s actions:\n"
            for index, action in enumerate(info[name]["action"]):
                action_str += f"{index+1}: {action}\n"
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": init_state_prompt.format(action_str)},
        ],
        model=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
        temperature=0.0
    )
    init_state = response.choices[0].message.content.strip()
    names = list(info.keys())
    if info[names[1]]["action"] is not None and info[names[1]]["utterance"] is None:
        prompt = f"""
        Consider the action of {names[1]} before {names[0]} act. Check where {names[1]} has put the object to help you determine {names[1]}'s desired location for the object.
        Actions: {info[names[1]]["action"]} 
        """ + latent_variable_prompt
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt.format(question)},
            ],
            model="gpt-4o",
            temperature=0.0
        )
    else:
        prompt = latent_variable_prompt + """
        State: {}
        """
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt.format(question, init_state)},
            ],
            model="gpt-4o",
            temperature=0.0
        )
    latent_variables = response.choices[0].message.content.strip()

    def extract_contents(label, input_string):
        pattern = rf'{label}: (.*?)(?=[A-Z]:|$)'
        match = re.search(pattern, input_string, re.DOTALL)
        return match.group(1).strip() if match else None

    a_contents = extract_contents('A', latent_variables)
    b_contents = extract_contents('B', latent_variables)
    c_contents = extract_contents('C', latent_variables)

    choices = {"A": a_contents, "B": b_contents, "C": c_contents}
    return init_state, choices

if __name__ == "__main__":
    pass