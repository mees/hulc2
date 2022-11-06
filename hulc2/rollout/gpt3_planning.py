import os

import openai

openai_key = os.environ["OPENAI_KEY"]
openai.api_key = openai_key

gpt3_prompt = """
state = {'drawer_open': False, 'blocks_on_table': ['red'], 'buttons_on': ['green']}
# put away the red block.
open_drawer()
pick_and_place('red', 'drawer')
close_drawer()
###
state = {'drawer_open': False, 'blocks_on_table': [], 'buttons_on': ['yellow']}
# turn off the lights.
push_button('yellow')
###
state = {'drawer_open': False, 'blocks_on_table': ['red', 'green', 'blue'], 'buttons_on': ['green', 'yellow']}
"""

gpt_version = "text-davinci-002"


def LM(prompt, max_tokens=128, temperature=0, stop=None):
    response = openai.Completion.create(
        engine=gpt_version, prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=stop
    )
    return response["choices"][0]["text"].strip()


user_input = "tidy up the workspace and turn off all the lights"
if user_input[-1] != ".":
    user_input += "."

context = gpt3_prompt
context += "# " + user_input + "\n"
response = LM(context, stop=["###"])
context += response + "\n"

step_text = ""


def pick_and_place(obj1, obj2):
    global step_text
    step_text = f"Pick the {obj1} block and place it on the {obj2}."


def open_drawer():
    global step_text
    step_text = "pull the handle to open the drawer"


def close_drawer():
    global step_text
    step_text = "pull the handle to close the drawer"


def push_button(obj1):
    global step_text
    if "green" in obj1:
        step_text = "press the button to turn on the led light"
    if "yellow" in obj1:
        step_text = "use the switch to turn on the light bulb"


# Execute commands given by LM.
step_cmds = response.split("\n")
print("LM generated plan:")
for step_cmd in step_cmds:
    step_cmd = step_cmd.replace("robot.", "")
    # print(step_cmd)
    exec(step_cmd)
    print("Step:", step_text)
    # obs = run_hucl(obs, step_text)
