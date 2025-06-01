"""
Chatbot Arena (battle) tab.
Users chat with two anonymous models.
"""

import json
import time
import re

import gradio as gr
import numpy as np

from fastchat.constants import (
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    SLOW_MODEL_MSG,
    BLIND_MODE_INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SURVEY_LINK,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.gradio_block_arena_named import flash_buttons
from fastchat.serve.gradio_web_server import (
    State,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    invisible_btn,
    enable_text,
    disable_text,
    acknowledgment_md,
    get_ip,
    get_model_description_md,
)
from fastchat.serve.remote_logger import get_remote_logger
from fastchat.utils import (
    build_logger,
    moderation_filter,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_sides = 2
enable_moderation = False
anony_names = ["", ""]
models = []


def set_global_vars_anony(enable_moderation_):
    global enable_moderation
    enable_moderation = enable_moderation_


def load_demo_side_by_side_anony(models_, url_params):
    global models
    models = models_

    states = [None] * num_sides
    selector_updates = [
        gr.Markdown(visible=True),
        gr.Markdown(visible=True),
    ]

    return states + selector_updates


def vote_last_response(states, vote_type, model_selectors, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "models": [x for x in model_selectors],
            "states": [x.dict() for x in states],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)

    gr.Info(
        "🎉 Thanks for voting! Your vote shapes the leaderboard, please vote RESPONSIBLY."
    )
    if ":" not in model_selectors[0]:
        for i in range(5):
            names = (
                "### Model A: " + states[0].model_name,
                "### Model B: " + states[1].model_name,
            )
            # yield names + ("",) + (disable_btn,) * 4
            yield names + (disable_text,) + (disable_btn,) * 5
            time.sleep(0.1)
    else:
        names = (
            "### Model A: " + states[0].model_name,
            "### Model B: " + states[1].model_name,
        )
        # yield names + ("",) + (disable_btn,) * 4
        yield names + (disable_text,) + (disable_btn,) * 5


def leftvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"leftvote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "leftvote", [model_selector0, model_selector1], request
    ):
        yield x


def rightvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"rightvote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "rightvote", [model_selector0, model_selector1], request
    ):
        yield x


def tievote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"tievote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    ):
        yield x


def bothbad_vote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"bothbad_vote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "bothbad_vote", [model_selector0, model_selector1], request
    ):
        yield x


def regenerate(state0, state1, request: gr.Request):
    logger.info(f"regenerate (anony). ip: {get_ip(request)}")
    states = [state0, state1]
    if state0.regen_support and state1.regen_support:
        for i in range(num_sides):
            states[i].conv.update_last_message(None)
        return (
            states + [x.to_gradio_chatbot() for x in states] + [""] + [disable_btn] * 6
        )
    states[0].skip_next = True
    states[1].skip_next = True
    return states + [x.to_gradio_chatbot() for x in states] + [""] + [no_change_btn] * 6


def clear_history(request: gr.Request):
    logger.info(f"clear_history (anony). ip: {get_ip(request)}")
    return (
        [None] * num_sides
        + [None] * num_sides
        + anony_names
        + [enable_text]
        + [invisible_btn] * 4
        + [disable_btn] * 2
        + [""]
        + [enable_btn]
    )


def share_click(state0, state1, model_selector0, model_selector1, request: gr.Request):
    logger.info(f"share (anony). ip: {get_ip(request)}")
    if state0 is not None and state1 is not None:
        vote_last_response(
            [state0, state1], "share", [model_selector0, model_selector1], request
        )


SAMPLING_WEIGHTS = {
    "gpt-3.5-turbo-0125": 1,
    "gpt-4.1-mini": 1, 
    "gemini-1.5-flash": 1,
    "gemini-2.0-flash": 1,
    "claude-3.5-haiku": 1,
    "claude-4-sonnet": 1
}

# target model sampling weights will be boosted.
BATTLE_TARGETS = {}

BATTLE_STRICT_TARGETS = {}

ANON_MODELS = []

SAMPLING_BOOST_MODELS = []

# outage models won't be sampled.
OUTAGE_MODELS = []


def get_sample_weight(model, outage_models, sampling_weights, sampling_boost_models=None):
    if sampling_boost_models is None:
        sampling_boost_models = []
        
    if not model or model in outage_models:
        return 0.0  # Explicitly return float 0.0 for models that should be excluded
    
    try:
        # Default weight is 1.0 if not in sampling_weights
        weight = float(sampling_weights.get(model, 1.0))
        
        # Ensure weight is a valid positive number
        if weight <= 0 or np.isnan(weight):
            weight = 1.0
            
        # Apply boost if model is in boost list
        if model in sampling_boost_models:
            weight = weight * 5.0
            
        return max(0.1, weight)  # Ensure we never return 0 to avoid division issues
    except Exception as e:
        print(f"Error calculating weight for model {model}: {e}")
        return 1.0  # Fallback to default weight


def is_model_match_pattern(model, patterns):
    flag = False
    for pattern in patterns:
        pattern = pattern.replace("*", ".*")
        if re.match(pattern, model) is not None:
            flag = True
            break
    return flag


def get_battle_pair(
    models, battle_targets, outage_models, sampling_weights, sampling_boost_models
):
    print(f"\n=== Starting get_battle_pair ===")
    print(f"Available models: {models}")
    print(f"Outage models: {outage_models}")
    print(f"Sampling weights: {sampling_weights}")
    print(f"Boost models: {sampling_boost_models}")
    
    if not models:
        raise ValueError("No models available for battle pairing")
        
    if len(models) == 1:
        print("Only one model available, using it for both sides")
        return models[0], models[0]

    model_weights = []
    for model in models:
        try:
            weight = get_sample_weight(
                model, outage_models, sampling_weights, sampling_boost_models
            )
            print(f"Model: {model}, Weight: {weight}")
            # Ensure weight is a valid number
            if not isinstance(weight, (int, float)) or np.isnan(weight) or weight < 0:
                print(f"Warning: Invalid weight for {model}, using default 1.0")
                weight = 1.0
            model_weights.append(float(weight))  # Ensure float type
        except Exception as e:
            print(f"Error getting weight for model {model}: {e}")
            model_weights.append(1.0)  # Fallback weight
    
    # Convert to numpy array for safe operations
    model_weights = np.array(model_weights, dtype=float)
    print(f"Raw weights: {model_weights}")
    
    # Replace any remaining NaN or negative weights with small positive value
    model_weights = np.where(
        (np.isnan(model_weights)) | (model_weights <= 0),
        0.1,  # Small positive weight
        model_weights
    )
    
    # If all weights are zero (shouldn't happen due to above checks), set uniform weights
    weight_sum = np.sum(model_weights)
    if weight_sum <= 0 or np.isnan(weight_sum):
        print("Warning: All weights are zero or invalid, using uniform distribution")
        model_weights = np.ones(len(models))
        weight_sum = len(models)
    
    # Normalize weights safely
    model_weights = model_weights / weight_sum
    print(f"Normalized weights: {model_weights}, sum: {np.sum(model_weights)}")
    
    # Final validation check
    if np.any(np.isnan(model_weights)) or abs(1.0 - np.sum(model_weights)) > 1e-6:
        print("Warning: Final weight validation failed, using uniform distribution")
        model_weights = np.ones(len(models)) / len(models)
    
    # Select model based on weights
    try:
        chosen_idx = np.random.choice(len(models), p=model_weights)
        chosen_model = models[chosen_idx]
        print(f"Selected model: {chosen_model} (index: {chosen_idx})")
    except Exception as e:
        print(f"Error in model selection: {e}, selecting first model as fallback")
        chosen_model = models[0] if models else None
    # for p, w in zip(models, model_weights):
    #     print(p, w)

    rival_models = []
    rival_weights = []
    for model in models:
        if model == chosen_model:
            continue
        if model in ANON_MODELS and chosen_model in ANON_MODELS:
            continue
        if chosen_model in BATTLE_STRICT_TARGETS:
            if not is_model_match_pattern(model, BATTLE_STRICT_TARGETS[chosen_model]):
                continue
        if model in BATTLE_STRICT_TARGETS:
            if not is_model_match_pattern(chosen_model, BATTLE_STRICT_TARGETS[model]):
                continue
        weight = get_sample_weight(model, outage_models, sampling_weights)
        if (
            weight != 0
            and chosen_model in battle_targets
            and model in battle_targets[chosen_model]
        ):
            # boost to 20% chance
            weight = 0.5 * total_weight / len(battle_targets[chosen_model])
        rival_models.append(model)
        rival_weights.append(weight)
    # for p, w in zip(rival_models, rival_weights):
    #     print(p, w)
    rival_weights = rival_weights / np.sum(rival_weights)
    rival_idx = np.random.choice(len(rival_models), p=rival_weights)
    rival_model = rival_models[rival_idx]

    swap = np.random.randint(2)
    if swap == 0:
        return chosen_model, rival_model
    else:
        return rival_model, chosen_model


def add_text(
    state0, state1, model_selector0, model_selector1, text, request: gr.Request
):
    ip = get_ip(request)
    logger.info(f"add_text (anony). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]

    # Init states if necessary
    if states[0] is None:
        assert states[1] is None

        model_left, model_right = get_battle_pair(
            models,
            BATTLE_TARGETS,
            OUTAGE_MODELS,
            SAMPLING_WEIGHTS,
            SAMPLING_BOOST_MODELS,
        )
        states = [
            State(model_left),
            State(model_right),
        ]

    if len(text) <= 0:
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + ["", None]
            + [
                no_change_btn,
            ]
            * 6
            + [""]
        )

    model_list = [states[i].model_name for i in range(num_sides)]
    # turn on moderation in battle mode
    all_conv_text_left = states[0].conv.get_prompt()
    all_conv_text_right = states[0].conv.get_prompt()
    all_conv_text = (
        all_conv_text_left[-1000:] + all_conv_text_right[-1000:] + "\nuser: " + text
    )
    flagged = moderation_filter(all_conv_text, model_list, do_moderation=True)
    if flagged:
        logger.info(f"violate moderation (anony). ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {get_ip(request)}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [CONVERSATION_LIMIT_MSG]
            + [
                no_change_btn,
            ]
            * 6
            + [""]
        )

    text = text[:BLIND_MODE_INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        states[i].conv.append_message(states[i].conv.roles[0], text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    hint_msg = ""
    for i in range(num_sides):
        if "deluxe" in states[i].model_name:
            hint_msg = SLOW_MODEL_MSG
    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + [""]
        + [
            disable_btn,
        ]
        * 6
        + [hint_msg]
    )


def bot_response_multi(
    state0,
    state1,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
):
    logger.info(f"bot_response_multi (anony). ip: {get_ip(request)}")

    if state0 is None or state0.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state0,
            state1,
            state0.to_gradio_chatbot(),
            state1.to_gradio_chatbot(),
        ) + (no_change_btn,) * 6
        return

    states = [state0, state1]
    gen = []
    for i in range(num_sides):
        gen.append(
            bot_response(
                states[i],
                temperature,
                top_p,
                max_new_tokens,
                request,
                apply_rate_limit=False,
                use_recommended_config=True,
            )
        )

    model_tpy = []
    for i in range(num_sides):
        token_per_yield = 1
        if states[i].model_name in [
            "gemini-pro",
            "gemma-1.1-2b-it",
            "gemma-1.1-7b-it",
            "phi-3-mini-4k-instruct",
            "phi-3-mini-128k-instruct",
            "snowflake-arctic-instruct",
        ]:
            token_per_yield = 30
        elif states[i].model_name in [
            "qwen-max-0428",
            "qwen-vl-max-0809",
            "qwen1.5-110b-chat",
            "llava-v1.6-34b",
        ]:
            token_per_yield = 7
        elif states[i].model_name in [
            "qwen2.5-72b-instruct",
            "qwen2-72b-instruct",
            "qwen-plus-0828",
            "qwen-max-0919",
            "llama-3.1-405b-instruct-bf16",
        ]:
            token_per_yield = 4
        model_tpy.append(token_per_yield)

    chatbots = [None] * num_sides
    iters = 0
    while True:
        stop = True
        iters += 1
        for i in range(num_sides):
            try:
                # yield fewer times if chunk size is larger
                if model_tpy[i] == 1 or (iters % model_tpy[i] == 1 or iters < 3):
                    ret = next(gen[i])
                    states[i], chatbots[i] = ret[0], ret[1]
                stop = False
            except StopIteration:
                pass
        yield states + chatbots + [disable_btn] * 6
        if stop:
            break


def build_side_by_side_ui_anony(models):
    # Using double curly braces to escape them in f-strings
    notice_markdown = """
# ⚔️  Stengo's Chatbot Arena: Compare & Test Best AI Chatbots

## 📜 How It Works
- **Blind Test**: Ask any question to two anonymous AI chatbots (currently includes ChatGPT, Claude, and Gemini).
- **Vote for the Best**: Choose the best response. You can keep chatting until you find a winner.
- **Play Fair**: If the AI's identity is revealed (especially if it is done intentionally), your vote won't count.

This is a custom version of [LMArena](https://lmarena.ai/), formerly known as Chatbot Arena and originally 
developed by researchers at UC Berkeley. LMArena/Chatbot Arena's underlying code is [FastChat](https://github.com/lm-sys/FastChat);
this chatbot arena uses FastChat too but with a few modifications.

## 📚 Education/Teaching-Focused
For this arena, we would like to focus on comparing the performance of AI chatbots in the context of **education and teaching** -
at this time, this is very general, so you may ask it to teach or tutor you in any topic you'd like (as long as it does not
violate the terms of service). This could be helping with math homework, learning a new programming language, figuring
out a complex physics or chemistry problem, diving deeper into a software engineering concept, and so on.

There are different ways you can evaluate the chatbots in this context:
- How well can it explain concepts to you as if you were a 5 year old, but without being too overly creative/simplified? 
- How helpful are the examples it provides where relevant? Are they too simple or overly complex? Relevant to orignal question?
- Can it provide an answer where you can follow its reasoning? Does it even provide its reasoning?
- Is it too verbose or did not provide enough details? How well does it do if you give it a vague prompt (e.g., "teach me calculus 1")?
- If you ask it to not reveal the answer to you and walk you through finding the solution by yourself, does it follow these instructions?

Any chats that are not focused on education/teaching will be marked as invalid and the votes will not count.

## 👇 Get started below!

<style>
#chatbot {
    height: 650px !important;
    overflow-y: auto !important;
    scroll-behavior: smooth !important;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 10px;
}

#chatbot .wrap {
    max-height: none !important;
}

#chatbot .message {
    margin-bottom: 15px;
    padding: 10px 15px;
    border-radius: 8px;
    line-height: 1.5;
}

#chatbot .user-message {
    background-color: #f5f5f5;
    margin-left: 20%;
    margin-right: 5px;
}

#chatbot .assistant-message {
    background-color: #e3f2fd;
    margin-right: 20%;
    margin-left: 5px;
}

#chatbot .message pre, #chatbot .message code {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
}
</style>
"""

    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots = [None] * num_sides

    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group(elem_id="share-region-anony"):
        with gr.Accordion(
            f"🔍 Expand to see the descriptions of {len(models)} models", open=False
        ):
            model_description_md = get_model_description_md(models)
            gr.Markdown(model_description_md, elem_id="model_description_markdown")
        with gr.Row():
            for i in range(num_sides):
                label = "Model A" if i == 0 else "Model B"
                with gr.Column():
                    chatbots[i] = gr.Chatbot(
                        label=label,
                        elem_id="chatbot",
                        height=650,
                        show_copy_button=True,
                        latex_delimiters=[
                            {"left": "$", "right": "$", "display": False},
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": r"\(", "right": r"\)", "display": False},
                            {"left": r"\[", "right": r"\]", "display": True},
                        ],
                        elem_classes=["chatbot-container"],
                    )

        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    model_selectors[i] = gr.Markdown(
                        anony_names[i], elem_id="model_selector_md"
                    )
        with gr.Row():
            slow_warning = gr.Markdown("")

    with gr.Row():
        leftvote_btn = gr.Button(
            value="👈  A is better", visible=False, interactive=False
        )
        rightvote_btn = gr.Button(
            value="👉  B is better", visible=False, interactive=False
        )
        tie_btn = gr.Button(value="🤝  Tie", visible=False, interactive=False)
        bothbad_btn = gr.Button(
            value="👎  Both are bad", visible=False, interactive=False
        )

    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your prompt and press ENTER",
            elem_id="input_box",
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row() as button_row:
        clear_btn = gr.Button(value="🎲 New Round", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        share_btn = gr.Button(value="📷  Share")

    with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=2048,
            value=2000,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    # Register listeners
    btn_list = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
        regenerate_btn,
        clear_btn,
    ]
    leftvote_btn.click(
        leftvote_last_response,
        states + model_selectors,
        model_selectors
        + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, send_btn],
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors,
        model_selectors
        + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, send_btn],
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors,
        model_selectors
        + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, send_btn],
    )
    bothbad_btn.click(
        bothbad_vote_last_response,
        states + model_selectors,
        model_selectors
        + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, send_btn],
    )
    regenerate_btn.click(
        regenerate, states, states + chatbots + [textbox] + btn_list
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )
    clear_btn.click(
        clear_history,
        None,
        states
        + chatbots
        + model_selectors
        + [textbox]
        + btn_list
        + [slow_warning]
        + [send_btn],
    )

    share_js = """
function (a, b, c, d) {
    const captureElement = document.querySelector('#share-region-anony');
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'chatbot-arena.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [a, b, c, d];
}
"""
    share_btn.click(share_click, states + model_selectors, [], js=share_js)

    textbox.submit(
        add_text,
        states + model_selectors + [textbox],
        states + chatbots + [textbox] + btn_list + [slow_warning],
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons,
        [],
        btn_list,
    )

    send_btn.click(
        add_text,
        states + model_selectors + [textbox],
        states + chatbots + [textbox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )

    return states + model_selectors
