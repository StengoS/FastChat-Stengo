Changes made to this version of FastChat:
* Out-of-the-box usage was not as well supported as expected or has unexpected errors that come up (especially those who have not developed much within the LLM ecosystem). I was able to fix how models would get randomly selected in the Arena (battle) version where the models are anonymized (set defaults to 1.0); from this, I also found out about how we'd usually have to set sampling weights in `gradio_block_arena_anony.py`.
* Made some content changes, primarily through removing stuff that would be relevant to the original developers and actual production version of the arena. I replaced contact points with my email.
* Added `api_endpoint.json` to be able to use the locally-deployed version of the arena with proprietary models through their respective APIs.
* Allowed for scrolling through the models' outputs in the frontend site's chat interface with the side-by-side comparisons.

I used Windsurf's SWE-1 to help with the following:
* How models get randomly selected the Arena (battle) version by setting defaults to be 1.0 instead.
* Scrolling down in the chat interface for the models' outputs that go beyond the chat interface.
* Adding a "Leaderboard" tab to the site (a reduced version of one from the original since I don't have the data that the orignal does).
* Quickly finding specific text displayed in the frontend to modify.
* Implement a feedback input box when voting for on models to collect qualitative data on *why* someone voted for a model over the other (took a lot more trial-and-error than expected; not yet able to figure out how to clear text from the box).

Current models in use (all through API usage):
- gpt-3.5-turbo
- gpt-4.1-mini
- gpt-4.1-nano
- o1-mini
- gemini-1.5-flash
- gemini-2.0-flash
- gemini-2.0-flash-lite
- gemini-2.5-flash-preview

Did try to use Claude models, but they were found to not be working with FastChat - most likely need to fix the original's implementation, but have not had a chance to do this yet. Did try with Windsurf's SWE-1, but was not successful. 

Other notes:
* Have to set up `api_endpoint.json` on your own, but very simple and there are enough instructions to get started.
* Using Gemini models will require you to install a separate package.
* If you plan on just evaluating proprietary models accessible through their respective APIs, you don't necessarily need a machine that can run entire models.
* Even if you have the OPENAI API key in `api_endpoint.json`, `api_provider.py` is broken and you need to set OPENAI_API_KEY in your env variables. Have not yet had a chance to fix it myself.
* If you run `basic_stats.py` and you get a ModuleNotFoundError, re-install FastChat using Method 2 ("From Source"). 
Referencing this https://github.com/lm-sys/FastChat/issues/302. 

Commands:
To start -
```
$ python3 -m fastchat.serve.gradio_web_server_multi --controller "" --share --register api_endpoint.json
```

To automatically tally votes and scores - 
```
$ python3 fastchat/serve/monitor/clean_battle_data.py [--max-num 10 --mode conv_release]
    - May need you to create a directory, "/home/[username]/fastchat_logs/server0", and set NUM_SERVERS in basic_stats.py to 1.
    - Copy your [date]-conv.json files into this new directory.

$ python3 fastchat/serve/monitor/elon_analysis.py --clean-battle-file [output from above command]
```

Other packages you may need - 
```
$ pip3 install google-generativeai (if you are using Gemini)
$ pip3 install polyglot pyicu pycld2
```

