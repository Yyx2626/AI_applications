# AI_applications
My demo applications using Artificial Intelligence (AI)

## yyx_OpenAI_prompt.20230331.py
I made this demo python program, because OpenAI API looks to be able to do more jobs than ChatGPT.  Specifically, OpenAI API includes whisper model to generate transcription from speech audio, and also include DALL-E for AI image generation.

My demo program mainly can call:
- OpenAI API
  - Chat, by gpt-3.5-turbo (by default)
  - Speech to text, by whisper
  - Image generation, by DALL-E
- gTTS (Google Text-to-Speech)
  - Text to speech (turned off, by default)

I utilized prompt_toolkit module to implement a simple terminal prompt user interface.

I originally planed to make the program that can allow users to directly chat with AI by audio. In principle, with its current abilities, it can work; however, due to low accuracy in speech recording and recognition (speech to text), it still seems a little far away from real application. I also implemented stop recording after detecting 1 second of silence, and implemented a function to suggest dB cutoff for 3-second silence, which will be automatically called when Setting Recorder.silence_dBFS_threshold .

BTW, sometimes I spoke English, whisper model will strangely return Chinese sentences with the same meaning. The issue due to multiple languages might be compromised by Setting AIcommunicator.whisper_languages to only one kind of language.

Note: some parts of the code were at first generated by ChatGPT (GPT-4.0 model). ChatGPT (GPT-4.0 model) can generate python code, which looks quite concise and elegant, but often buggy. Then, I have to spend time debugging the code. Anyway, ChatGPT (GPT-4.0 model) can be a great tool to facilitate coding.

### Usage

After downloading `yyx_OpenAI_prompt.20230331.py`, you need to manually modify **Line 70** by pasting your **OpenAI API key** in ''

Note: You can sign up OpenAI API at https://platform.openai.com/signup , and create OpenAI API key at https://platform.openai.com/account/api-keys .
New user may get some dollars granted by OpenAI for free trial.

Then, you can execute `python yyx_OpenAI_prompt.20230331.py` in command line, such as Windows PowerShell.

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

2023-03-31, I also added the function to automatically appending a log file with date (format %Y%m%d) in the filename, such as 'yyx_OpenAI_prompt.20230331.log'.
It contains all the output to screen except for the help page and Setting process, and with no colors.

### Limitation and possible future plan

1. I originally planned to implement recording microphone when I keep pressing some keys (such as F8), and also attemped to exit the Setting menus by pressing Esc. There are some modules that may monitor keyboard events; however, I have not figured out the way to notify the recorder to stop recording when an event is detected. This may also be related to the following point.
2. I attempted to put audio processing (recording, segmentating and playing) into some parallel threads, and also attempted to processing data in stream. However, it became complicated and generated weird behavior. Therefore, for simplicity, current version of the program only has one process and one thread; thus, recording will block all other responses.
3. I have not got access to GPT-4 model, which is supposed to be able to process both text and image. Currently, ChatGPT Plus users can access GPT-4 model in ChatGPT (https://chat.openai.com/chat?model=gpt-4); however, it seems that ChatGPT's input is limited to text. So after I can access GPT-4 model, I may try to also implement image input.

### Version and change logs

- 20230328: initially upload yyx_OpenAI_prompt.20230328.py

- 20230331: update to yyx_OpenAI_prompt.20230331.py
  - remove temporary audio files if exception (including Ctrl-c) occurs
  - add appending log file
