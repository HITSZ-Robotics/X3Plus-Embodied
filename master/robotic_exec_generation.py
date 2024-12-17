import os
import openai
import json
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"


script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(script_path)
base_path=parent_directory+"/data/"
folder =base_path
chat_path=base_path+"chat.json"
# Load different types of prompt
system_file = "system_prompt_file.ini"

with open(system_file, "r") as f:
    system_prompt_file = f.readlines()
system_prompt_file = "".join(system_prompt_file)

openai.api_key = ''
# This API key is only for demo, please use your own API key
prompt_style = "VISPROG"

# You can choose different prompt here
# visual_programm_prompt: VISPROG style prompt
# full_prompt_i2a: VISPROG + ViperGPT style prompt


if not os.path.exists(folder):
    os.makedirs(folder)

def turn_list_to_string(all_result):
    if not isinstance(all_result, list):
        return all_result
    all_in_one_str = ""
    for r in all_result:
        all_in_one_str = all_in_one_str + r + "\n"
    if all_in_one_str.endswith("\n"):
        all_in_one_str = all_in_one_str[:-1]
    if all_in_one_str.endswith("."):
        all_in_one_str = all_in_one_str[:-1]
    return all_in_one_str

def result_preprocess(results):
    """
        Only used for the result with full_prompt_i2a
    """
    codes = []
    if isinstance(results, list):
        results = results[0]
    for code in results.splitlines():
        if "main" in code :
            continue
        if code.startswith("#"):
            continue
        if "```" in code :
            continue
        codes.append(code.strip())
    
    return codes

def insert_task_into_prompt(task, prompt_base, insert_index="INSERT TASK HERE"):
    full_prompt = prompt_base.replace(insert_index, task)
    return full_prompt

def exec_steps(user_content):
    save_file = "exec_code.txt"
    
    save_file = os.path.join(folder, save_file)

    
    #if os.path.exists(save_file) and reuse:
    #    # not needed exactly, just because the cost and time of openai api
    #    print(f"The code file already exists, direct load {save_file}")
    #    with open(save_file, "r") as tfile:
    #        all_result = tfile.readlines()
    #        all_in_one_str = turn_list_to_string(all_result)
    #        return all_in_one_str
    #else:
    if os.path.exists(chat_path):
        with open(chat_path,"r")as f:
            messages=json.load(f)
    else:
        messages = [{"role": "system", "content": system_prompt_file}]
        
    messages.append({"role": "user", "content": user_content})
    trials = 0
        # the response could be incomplete, so we need to run it multiple times
    while True and trials < 5:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.99,
            max_tokens=256,
            n=1,
            
        )
        all_result = []
        for r in range(len(response.choices)):
           result = response.choices[r].message.content
           #print(response.choices[r].message)
           if prompt_style == "instruct2act":
               all_result.append(result)
           else:
               all_result.append(result.replace("\n\n", ""))
        
        all_result = result_preprocess(all_result)
        messages.append({"role":"assistant","content":response.choices[0].message.content})
        trials += 1
        if len(all_result) > 0: # the result should be at least 5 lines code
            break
        else:
            print("The result is too short, retry...")
            print(all_result)
            continue
    print("Save result to: ", save_file)
    with open(save_file, "w") as tfile:
        tfile.write("\n".join(all_result))
    with open(chat_path,"w") as f:
         json.dump(messages,f) 
    all_result = turn_list_to_string(all_result)
    return all_result




