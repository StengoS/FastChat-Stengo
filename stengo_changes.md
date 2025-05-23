Changes to made to this version of FastChat:
* Out-of-the-box usage is not well supported or has unexpected errors that come up (especially those who have not developed much within the LLM ecosystem). I was able to fix how models would get randomly selected in the Arena (battle) version where the models are anonymized (set defaults to 1.0); from this, I also found out about how we'd usually have to set sampling weights in `gradio_block_arena_anony.py`.
* Made some content changes, primarily through removing stuff that would be relevant to the original developers and actual production version of the arena. I replaced contact points with my email.
* Added `api_endpoint.json` to be able to use the locally-deployed version of the arena with proprietary models through their respective APIs.
* Allowed for scrolling through the models' outputs in the frontend site's chat interface with the side-by-side comparisons.

Windsurf helped with:
* How models get randomly selected the Arena (battle) version by setting defaults to be 1.0 instead.
* Scrolling down in the chat interface for the models' outputs that go beyond the chat interface.

Current models in use:
- gpt-3.5-turbo
- gpt-4.1-mini
- gemini-1.5-flash
- gemini-2.0-flash

Other notes:
* Have to set up `api_endpoint.json` on your own, but very simple and there are enough instructions to get started.
* Using Gemini models will require you to install a separate package.
* If you plan on just evaluating proprietary models accessible through their respective APIs, you don't necessarily need a machine that can run entire models.

Commands:
To start -
```
$ python3 -m fastchat.serve.gradio_web_server_multi --controller "" --share --register api_endpoint.json
```


TODOs:
1. Get this onto a private GitHub repo, deploy it on the Cyber@UCI infra, and then set up a reverse proxy to make it publicly accessible.
2. Automate collecting usage telemetry.