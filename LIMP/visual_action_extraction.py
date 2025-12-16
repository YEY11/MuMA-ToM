from openai import OpenAI
import json
import ast
import re
import os
from typing import List
from dotenv import load_dotenv
load_dotenv()
# LIMP 视觉动作摘要模块：从 Files/actions_extracted.json 和文本生成规整的动作序列
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)
def get_action(episode_id: int) -> List[str]:
    """读取指定 episode 的原始动作与文本，抽取并规整为动作列表
    参数:
        episode_id (int): 目标 episode 标识
    返回:
        List[str]: 规整后的动作序列
    """
    with open("../Files/actions_extracted.json", "r") as file:
        data = json.load(file)
    with open("../Files/texts.json", "r") as file:
        text = json.load(file)[str(episode_id)]
    text = text.split("\n")[0]
    name = text.split(":")[0]
    if episode_id < 4000:
        prompt = f"""
            Read a piece of text, select the object that the person is picking up and moving around, only include the object name in your answer.

            Text: {text}
        """
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt.format(data[str(episode_id)]["action"], text, name)},
            ],
            model=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
            logprobs=True,
            top_logprobs=5,
            temperature=0.0
        )
        text = response.choices[0].message.content.strip()
        print(text)

        prompt = f"""Read a piece of text, select a person's name from the text. Only output person's name
        Input text: {data[str(episode_id)]["action"]}
        """
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt.format(data[str(episode_id)]["action"], text, name)},
            ],
            model=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
            logprobs=True,
            top_logprobs=5,
            temperature=0.0
        )
        name = response.choices[0].message.content.strip()
        print(name)
    prompt = """
    Input text: {}
    Additional_information: {}
    Person's name: {}
    You will read some text describe a person's action. The name of the person is given. Only summarize his/her action and ignore actions of other person. Reorganize the person's actions.
    Possible actions include: walk towards somewhere, grab something from somewhere, open some container, close some container, put something somewhere. Only summarize these actions and their synonyms in this form and abandon mismatch actions. Omit peron's name. When mentioning location name, try to infer room the location is inside and include it in the action in form "[container] in [room_name]"
    Check objects mentioned in the Additional Information section. Replace any object mentioned in action with the object appeared in that section
    Formulate your final answer in the following form.
    Actions:
    ["action1", "action2", ....]
    """  # 规整动作的提示词：仅保留关键动作类型并统一格式
    print(prompt.format(data[str(episode_id)]["action"], text, name))
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt.format(data[str(episode_id)]["action"], text, name)},
        ],
        model=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
        logprobs=True,
        top_logprobs=5,
        temperature=0.0
    )
    actions = response.choices[0].message.content.strip()
    actions_match = re.search(r'Actions:\s*(\[[^\]]*\])', actions)
    action = actions_match.group(1) if actions_match else None
    action_prediction = ast.literal_eval(action)
    print(action_prediction)
    return action_prediction