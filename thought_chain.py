import os
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, List
from response_format.prompt import format_prompt_getter
from response_format.parser import parse_model_to_json
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="****************************************************************",
    base_url="https://openrouter.ai/api/v1"
)

class STEPS_FLOW(BaseModel):
    steps: str = Field(description="实施步骤")
    explaination: Optional[str] = Field(description="实施原因")

@format_prompt_getter
class ANSRESULT(BaseModel):
    ans: List[STEPS_FLOW] = Field(description="最终结果")


def PRM_eval(question,steps):
    payload = {
        "problem": (question),
        "response": (steps)
    }
    resp = requests.post("https://lihongze8-RM.hf.space/api/predict", json=payload)
    data = resp.json()
    return data

# 创建函数 get_thought_steps
def get_thought_steps(prompt: str) -> str:
    json_text, json_obj = parse_model_to_json(ANSRESULT)

    # 调用 OpenAI 客户端
    response = client.chat.completions.create(
        model="qwen/qwen-2-7b-instruct:free",
        messages=[
            {"role": "system", "content": f"{ANSRESULT.get_response_format_prompt()}"},
            {"role": "user", "content": f"{prompt}"}
        ],
        top_p=0.7,
        temperature=0.1,
        max_tokens=4000
    )
    step_ans = response.choices[0].message.content
    return step_ans

def get_final_ans(prompt: str,cot: str) -> str:
    # 调用 OpenAI 客户端
    response = client.chat.completions.create(
        model="qwen/qwen-2-7b-instruct:free",
        messages=[
            {"role": "user", "content": f"{prompt}follow those steps{cot}"}
        ],
        top_p=0.7,
        temperature=0.7
    )
    final_ans = response.choices[0].message.content
    return final_ans

# 创建函数 batch_thought_chain
def batch_thought_chain(prompt: str, N: int) -> List[str]:
    results = []
    # 使用线程池并发执行 N 次 get_thought_steps
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_thought_steps, prompt) for _ in range(N)]
        for future in as_completed(futures):
            results.append(future.result())
    return results

