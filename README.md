# AI_applications
My demo applications using Artificial Intelligence (AI)

## yyx_OpenAI_prompt.20230328.py
I made this demo python program, which mainly can call:
- OpenAI
  - Chat, by gpt-3.5-turbo (by default)
  - Speech to text, by whisper
  - Image generation, by DALL-E
- gTTS (Google Text-to-Speech)
  - Text to speech (turned off, by default)

It has a simple user interface implemented by prompt_toolkit.

### Usage

After downloading `yyx_OpenAI_prompt.20230328.py`, you need to manually modify Line 69 by pasting your OpenAI API key in ''

Note: You can sign up OpenAI API at https://platform.openai.com/signup, and create OpenAI API key at https://platform.openai.com/account/api-keys 

Then, you can execute `python yyx_OpenAI_prompt.20230328.py` in command line, such as Windows PowerShell.

It may first prompt to install modules (by pip) if some are missing.

Then, please wait for several seconds to allow it load the modules.

After that, it will prompt the simple help page as follows, and prompt '> ' to wait for your input.

```
Type  'exit' and Enter  or  Ctrl-c  to end program
Type  'Setting' and Enter  to configure settings
Type  'L' and Enter  to start recording microphone
    (automatically stop after 1 sec of silence)
Type or say  'Please draw picture of ...' or 'Can you draw picture of ...' or '请画图...'  to ask DALL-E draw a image
Type or say  'Please draw picture again' or '请重画'  to ask DALL-E draw image with previous prompt again
```

DALL-E generated images will be downloaded and saved in current directory with the datetime stamp (format %Y%m%d%H%M%S) in the filename, such as 'image.20230328091510.jpg'.

And after calling OpenAI API, it may estimate the cost according to https://openai.com/pricing

It will be summarized and saved in current directory with date (format %Y%m%d) in the filename, such as 'total_cost.20230328.txt'.

The total_cost txt file will contain 3 columns:
 1. datetime stamp (format %Y%m%d%H%M%S) when the program starts
 2. total estimated cost in dollars ($)
 3. item details for the estimation

### Limitation and Possible Future Plan

1. I originally planned to implement recording microphone when I keep pressing some keys (such as F8), and also attemped to exit the Setting menus by pressing Esc. There are some modules that may monitor keyboard events; however, I have not figured out the way to notify the recorder to stop recording when an event is detected. This may also be related to the following point.
2. I attempted to put audio processing (recording, segmentating and playing) into some parallel threads, and also attempted to processing data in stream. However, it became complicated and generated weird behavior. Therefore, for simplicity, current version of the program only has one process and one thread; thus, recording will block all other responses.
