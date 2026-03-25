import  subprocess
import re

# ---------------------------------------
# OpenAI API 调用
# ---------------------------------------
def generate_with_api(client, prompt: str, modelname: str,completion_kwargs:dict):
    assert modelname in ["gemini-2.0-flash", 'gemini-2.5-flash', 'gpt-4o','deepseek-v3','gpt-4o-2024-11-20','grok-3','gemini-2.0-flash-thinking-exp']
    messages = [{"role": "user", "content": prompt}]
    if(modelname=='deepseek-v3'):
        modelname = 'deepseek-chat'
    response = client.chat.completions.create(
            messages=messages,
            model=modelname,
            **completion_kwargs
        )
    # print(response)  # 移除打印，保存到文件
    result_text = str(response.choices[0].message.content)
    return result_text

##提取代码块
def extract_code_block(llm_output: str) -> str:
    """
    使用正则提取三引号 ```python ...``` 之间的代码（DOTALL 模式）。
    若未匹配到则返回空字符串。
    """
    pattern = r'<python>(.*?)</python>'
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if '```' in code: #可能python内部额外加了代码块
            pattern = r'```python(.*?)```'
            match = re.search(pattern, code, re.DOTALL)
            if match:
                code = match.group(1).strip()
        return code
    # 可能没有pyhon符号
    pattern = r'```python(.*?)```'
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        return code
    return None

## code是完整的原始代码
def execute_code_output(code):
    try:
        result = subprocess.run(['python', '-c', code], capture_output=True, text=True, timeout=100)  # 设置超时时间为10秒
        output = result.stdout
        error = result.stderr
        return output, error
    except subprocess.TimeoutExpired:
        return None, "Execution timed out"
    except Exception as e:
        return None, str(e)