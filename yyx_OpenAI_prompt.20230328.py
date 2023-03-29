
### ref: ChatGPT (model: GPT-4.0)

import importlib
import subprocess
import sys

def install_package(package_name):
    print(f'Package {package_name} is not installed. Do you want me to install it (by pip)? (Yes/No) ', end='')
    user_input = input().lower()

    if user_input == 'yes' or user_input == 'y':
        try:
            print(f'Now installing {package_name}...')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
            print(f'Package {package_name} is installed successfully')
            return 1
        except subprocess.CalledProcessError as e:
            print(f'Error in installing {package_name}: {e}')
            return -1
    else:
        print(f'Package {package_name} is not installed')
        return 0

def ensure_package_installed(package_name):
    try:
        importlib.import_module(package_name)
    except ImportError:
        ret = install_package(package_name)
        if ret <= 0:
            sys.exit(f'Error: Package {package_name} is not installed')

ensure_package_installed('prompt_toolkit')
ensure_package_installed('sounddevice')
ensure_package_installed('numpy')
ensure_package_installed('pydub')
ensure_package_installed('soundfile')
ensure_package_installed('openai')
ensure_package_installed('gtts')
ensure_package_installed('requests')
ensure_package_installed('librosa')




### ref: ChatGPT (model: GPT-4.0)

from prompt_toolkit import PromptSession
from prompt_toolkit import print_formatted_text, prompt
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.validation import Validator, ValidationError

import sounddevice   # ref: https://python-sounddevice.readthedocs.io/en/0.3.14/examples.html
import numpy as np
import pydub
import threading
import itertools
import soundfile
import datetime
import time
import openai
import gtts
import os
import re
import requests
import librosa
import statistics

openai.api_key = ''
## Note: You can sign up OpenAI API on https://platform.openai.com/signup
##       Then, you can create OpenAI API key on https://platform.openai.com/account/api-keys


### flushing print, reference: https://mail.python.org/pipermail/python-list/2015-November/698426.html
def _print(*args, **kwargs):
    file = kwargs.get('file', sys.stdout)
    print(*args, **kwargs)
    file.flush()


### reference: Yyx_system_command_functions.20160607.pl


def check_elapsed_time(start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    day = int(elapsed_time / (3600*24))
    hour = int(elapsed_time % (3600*24) / 3600)
    min = int(elapsed_time % 3600 / 60)
    sec = elapsed_time % 60
    elapsed_time = ''
    if day>0 : elapsed_time += '{}day '.format(day)
    if hour>0: elapsed_time += '{}h'.format(hour)
    if min>0 : elapsed_time += '{}min'.format(min)
    if sec>0 or elapsed_time == '': elapsed_time += '{:.2f}s'.format(sec)
    _print('[PYTHON-TIME] ' + elapsed_time, file=sys.stderr)


### ref: https://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
def strtobool(val, default=None):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        if default is not None:
            _print(f'ValueError: invalid boolean value {val}, so I return default value {default}')
            return default
        else:
            raise ValueError(f'invalid boolean value {val}')


### ref: https://python-prompt-toolkit.readthedocs.io/en/master/pages/asking_for_input.html#asking-for-input
class NumberValidator(Validator):
    def validate(self, document):
        text = document.text
        if text and not text.isdigit():
            i = 0
            # Get index of first non numeric character.
            # We want to move the cursor here.
            for i, c in enumerate(text):
                if not c.isdigit():
                    break
            raise ValidationError(message='This input contains non-numeric characters',
                                  cursor_position=i)

#esc_bindings = KeyBindings()
#
#@esc_bindings.add('escape')
#def _(event):
#    ### ref: https://stackoverflow.com/questions/59675404/adding-a-key-binding-on-python-prompt-toolkit-3-0-2-breaks-the-suggestion-and-hi
#    event.app.exit(result='0')
#    ## Note: this works, but respond quite slowly

class Setting:
    prompt_allow_multiline = False
    
    @staticmethod
    def show_setting_list_and_prompt(key_value_list):
        N = len(key_value_list)
        for i in range(1, N+1):
            print(f'{i} :\t{key_value_list[i-1][0]}', end='')
            if len(key_value_list[i-1]) > 1:
                print(f' = \t{key_value_list[i-1][1]}')
            else:
                print()
        try:
            idx = prompt('\nChoose which item (0 or Ctrl-c for exit)? ', validator=NumberValidator())   #, key_bindings=esc_bindings)
            idx = int(idx)
            if idx <= 0:
                return 0
            else:
                return idx
        except KeyboardInterrupt:
            return 0
        except:
            return -1
    
    @staticmethod
    def show_setting():
        global total_est_cost, total_cost_items
        [total_est_cost_str, total_cost_item_str] = generate_total_cost_strings(total_est_cost, total_cost_items)
        _print(f'Total estimated cost so far is ${total_est_cost_str}  for {total_cost_item_str}')
        idx = -1
        while idx < 0:
            _print()
            idx = Setting.show_setting_list_and_prompt([['Recorder'], ['AIcommunicator'], ['prompt_allow_multiline', Setting.prompt_allow_multiline]])
            if idx == 0:
                return
            elif idx == 1:   # Recorder
                Setting.show_Recorder_setting()
            elif idx == 2:   # AIcommunicator
                Setting.show_AIcommunicator_setting()
            elif idx == 3:   # prompt_allow_multiline
                Setting.prompt_allow_multiline = not Setting.prompt_allow_multiline
                if Setting.prompt_allow_multiline:
                    _print('Note: you need to use  Esc + Enter  to submit in multiline mode')
            elif idx != -1:
                print_formatted_text(FormattedText([('red', f'Error: Unrecognized choice {idx}')]))
            idx = -1


    @staticmethod
    def get_Recorder_setting():
        keys = ['sample_rate', 'channels', 'dtype', 'silence_dBFS_threshold', 'silence_time_sec_threshold']
        return [[x , getattr(Recorder, x)] for x in keys]
    
    @staticmethod
    def set_Recorder_setting(dict):
        str_keys = ['dtype']
        int_keys = ['sample_rate', 'channels', 'silence_dBFS_threshold']
        float_keys = ['silence_time_sec_threshold']
        for key, value in dict.items():
            if key in str_keys:
                setattr(Recorder, key, value)
            if key in int_keys:
                setattr(Recorder, key, int(value))
            if key in float_keys:
                setattr(AIcommunicator, key, float(value))

    @staticmethod
    def show_Recorder_setting():
        idx = -1
        while idx < 0:
            prev_setting = Setting.get_Recorder_setting()
            idx = Setting.show_setting_list_and_prompt(prev_setting)
            if idx == 0:
                return
            elif idx > 0 and idx <= len(prev_setting):
                if prev_setting[idx-1][0] == 'silence_dBFS_threshold':
                    suggested_dBFS_threshold, avg_dBFS_vec = Recorder.suggest_dBFS_threshold()
                    _print(f'Suggested dBFS threshold {suggested_dBFS_threshold}, according to ' + ', '.join(map(str, avg_dBFS_vec)))
                try:
                    new_value = prompt(f'\nSet {prev_setting[idx-1][0]} new value: ', default=f'{prev_setting[idx-1][1]}')
                    Setting.set_Recorder_setting({ prev_setting[idx-1][0] : new_value })
                except KeyboardInterrupt:
                    idx = -1
                    continue
                except Exception as e:
                    print_formatted_text(FormattedText([('red', f'Error: {e}')]))
                    idx = -1
                    continue
            elif idx != -1:
                print_formatted_text(FormattedText([('red', f'Error: Unrecognized choice {idx}')]))
            idx = -1

    @staticmethod
    def get_AIcommunicator_setting():
        keys = ['should_show_elapsed_time', 'should_show_cost', 'image_size', 'speech_filename', 'whisper_languages', 'gtts_speed_ratio', 'chat_model', 'should_speak']
        return [[x , getattr(AIcommunicator, x)] for x in keys]
    
    @staticmethod
    def set_AIcommunicator_setting(dict):
        str_keys = ['speech_filename', 'whisper_languages', 'chat_model']
        int_keys = ['image_size']
        float_keys = ['gtts_speed_ratio']
        bool_keys = ['should_show_elapsed_time', 'should_show_cost', 'should_speak']
        for key, value in dict.items():
            if key in str_keys:
                if key == 'chat_model':
                    if value not in AIcommunicator.list_models():
                        print_formatted_text(FormattedText([('red', f'Error: Unrecognized model {value}')]))
                        return
                setattr(AIcommunicator, key, value)
            if key in int_keys:
                if key == 'image_size':
                    if value not in [256, 512, 1024]:
                        print_formatted_text(FormattedText([('red', f'Error: image_size should be one of 256, 512 or 1024')]))
                        return
                setattr(AIcommunicator, key, int(value))
            if key in float_keys:
                setattr(AIcommunicator, key, float(value))
            if key in bool_keys:
                setattr(AIcommunicator, key, strtobool(value))

    @staticmethod
    def show_AIcommunicator_setting():
        bool_keys = ['should_show_elapsed_time', 'should_show_cost', 'should_speak']
        idx = -1
        while idx < 0:
            prev_setting = Setting.get_AIcommunicator_setting()
            idx = Setting.show_setting_list_and_prompt(prev_setting)
            if idx == 0:
                return
            elif idx > 0 and idx <= len(prev_setting):
                key = prev_setting[idx-1][0]
                if key in bool_keys:
                    setattr(AIcommunicator, key, not getattr(AIcommunicator, key))
                else:
                    if key == 'chat_model':
                        _print('\nCurrently available OpenAI models: ' + ', '.join(AIcommunicator.list_models()))
                        _print('\nWarning: most models do not work with chat code here')
                    try:
                        new_value = prompt(f'\nSet {key} new value: ', default=f'{prev_setting[idx-1][1]}')
                        Setting.set_AIcommunicator_setting({ key : new_value })
                    except KeyboardInterrupt:
                        idx = -1
                        continue
                    except Exception as e:
                        print_formatted_text(FormattedText([('red', f'Error: {e}')]))
                        idx = -1
                        continue
            elif idx != -1:
                print_formatted_text(FormattedText([('red', f'Error: Unrecognized choice {idx}')]))
            idx = -1


class Recorder:
    sample_rate = 16000
    channels = 1
    dtype = 'int16'
    silence_dBFS_threshold = -60
    silence_time_sec_threshold = 1 #sec
    
    def __init__(self):
        self.sample_rate = Recorder.sample_rate
        self.channels = Recorder.channels
        self.dtype = Recorder.dtype
        self.silence_dBFS_threshold = Recorder.silence_dBFS_threshold
        self.silence_time_sec_threshold = Recorder.silence_time_sec_threshold
        self.should_stop = False
        self.silence_start_time = -1
        self.audio_buffer = []
        
    def audio_callback(self, indata, frames, time, status):
        self.audio_buffer.extend(indata.copy())
        currentTime = len(self.audio_buffer) / self.sample_rate
#        print(f'frames = {frames} , time = {time} , status = {status}')
        audio_segment = pydub.AudioSegment(
            indata.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=indata.dtype.itemsize,
            channels=self.channels
        )
#        print(audio_segment.dBFS)
        if audio_segment.dBFS < self.silence_dBFS_threshold:
#            print(currentTime)
            if self.silence_start_time < 0:
                self.silence_start_time = currentTime
            elif currentTime - self.silence_start_time > self.silence_time_sec_threshold:
                self.should_stop = True
        else:
            self.silence_start_time = -1

    def start_recording(self, output_filename):
        with sounddevice.InputStream(samplerate=self.sample_rate, channels=self.channels, dtype=self.dtype, callback=self.audio_callback):
            while not self.should_stop:
                sounddevice.sleep(1000)
        
        _print(f'Recording time: {len(self.audio_buffer)/self.sample_rate} sec')
        soundfile.write(output_filename, self.audio_buffer, self.sample_rate)
        
        self.should_stop = False
        self.silence_count = 0
        self.audio_buffer = []
    
    @staticmethod
    def record_to(output_filename):
        recorder = Recorder()
        recorder.start_recording(output_filename)

    @staticmethod
    def suggest_dBFS_threshold(duration=3):
        _print(f'\nKeep silence for {duration} sec, now testing silence dB ...')
        audio_data = sounddevice.rec(int(duration * Recorder.sample_rate), samplerate=Recorder.sample_rate, channels=1, dtype=Recorder.dtype)
        sounddevice.wait()
        audio_segment = pydub.AudioSegment(
            audio_data.tobytes(),
            frame_rate=Recorder.sample_rate,
            sample_width=audio_data.dtype.itemsize,
            channels=Recorder.channels
        )
        avg_dBFS_vec = []
        for x in range(1, duration*2):
            now_dBFS = audio_segment[x*500:(x+1)*500].dBFS
            avg_dBFS_vec.append(now_dBFS)
        _print(avg_dBFS_vec)
        return round(max(avg_dBFS_vec) + statistics.stdev(avg_dBFS_vec) * 3 + 2), avg_dBFS_vec


chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')

total_est_cost = 0
total_cost_items = {}
program_datetime_stamp = datetime.datetime.now()

def generate_total_cost_strings(total_est_cost, total_cost_items):
    total_est_cost_str = np.format_float_positional(round(total_est_cost, 7))
    total_cost_item_str = ''
    for key in sorted(total_cost_items.keys()):
        total_cost_item_str += f' + {total_cost_items[key]} {key}'
    total_cost_item_str = total_cost_item_str[3:]
    return [total_est_cost_str, total_cost_item_str]

def update_total_est_cost_file(total_est_cost, datetime_stamp, total_cost_items):
    date_str = datetime_stamp.strftime('%Y%m%d')
    datetime_str = datetime_stamp.strftime('%Y%m%d%H%M%S')
    [total_est_cost_str, total_cost_item_str] = generate_total_cost_strings(total_est_cost, total_cost_items)
    newF = [datetime_str, total_est_cost_str, total_cost_item_str]
    total_est_cost_filename = 'total_cost.' + date_str + '.txt'
    lines = []
    if os.path.exists(total_est_cost_filename) and os.path.isfile(total_est_cost_filename):
        with open(total_est_cost_filename, 'r') as fin:
            lines = fin.readlines()
    with open(total_est_cost_filename, 'w') as fout:
        has_found = False
        for line in lines:
            line = line.rstrip('\r\n')
            F = line.split('\t')
            if F[0] == '':
                continue
            if F[0] == datetime_str:
                F = newF
                has_found = True
            _print('\t'.join(F), file=fout)
        if not has_found:
            _print('\t'.join(newF), file=fout)


class AIcommunicator:
    should_show_elapsed_time = True
    should_show_cost = True
    image_size = 512
    speech_filename = 'speech.wav'
    whisper_languages = 'en,zh'
    gtts_speed_ratio = 1.4
    chat_model = 'gpt-3.5-turbo'
    should_speak = False
    OpenAI_models = []
    
    @classmethod
    def recognize_speech_whisper(cls, audio_filename):
        global total_est_cost, total_cost_items
        languages = cls.whisper_languages.split(',')
        audio_data, sample_rate = soundfile.read(audio_filename)
        audio_sec_length = round(len(audio_data) / sample_rate)
        est_cost = round(audio_sec_length * 0.006/60, 4)
        total_est_cost += est_cost
        total_est_cost = round(total_est_cost, 7)
        if 'whisper' not in total_cost_items:
            total_cost_items['whisper'] = 0
        total_cost_items['whisper'] += audio_sec_length
        update_total_est_cost_file(total_est_cost, program_datetime_stamp, total_cost_items)
        if cls.should_show_cost:
            _print(f'Estimated cost ${np.format_float_positional(est_cost)} for {audio_sec_length} sec input')
        with open(audio_filename, 'rb') as faudio:
            start_time = time.time()
            response = openai.Audio.transcribe('whisper-1', faudio, language=languages)
            if cls.should_show_elapsed_time:
                check_elapsed_time(start_time)   # 1.19s
#            print(response)
            return response['text']

    @staticmethod
    def play_faster(audio_data, sample_rate, speed_ratio):
        stretched_audio_data = librosa.effects.time_stretch(audio_data, rate=speed_ratio)
        sounddevice.play(stretched_audio_data, sample_rate)
    
    @classmethod
    def gtts_speak(cls, text):
        language = 'en'
        if chinese_char_pattern.search(text):
            language = 'zh'
        # Use gTTS to convert text to speech
        start_time = time.time()
        speech = gtts.gTTS(text=text, lang=language, slow=False)
        if cls.should_show_elapsed_time:
            check_elapsed_time(start_time)   # 1.19s
        # Note: gtts can only support one language at one time   # ref: https://stackoverflow.com/questions/70852444/how-to-use-muti-language-in-gtts-for-single-input-line

        # Save speech as a WAV file
        speech.save(cls.speech_filename)

        # Load the WAV file as a NumPy array
        audio_data, sample_rate = soundfile.read(cls.speech_filename)

        # Play the NumPy array through the PC speaker using the sounddevice library
        AIcommunicator.play_faster(audio_data, sample_rate, cls.gtts_speed_ratio)
#        sounddevice.play(audio_data, sample_rate)
#        sounddevice.wait()
        
        os.remove(cls.speech_filename)


    @classmethod
    def moderate(cls, input):
        moderation_resp = openai.Moderation.create(input=input)
#        _print(moderation_resp)
        moderation_resp = moderation_resp['results'][0]
        if moderation_resp.flagged:
            max_key = ''
            max_value = 0
            for key, value in moderation_resp.category_scores.items():
                if value > max_value:
                    max_key, max_value = key, value
            return max_key
        else:
            return ''

    @classmethod
    def chat(cls, content, role='user'):
        global total_est_cost
        start_time = time.time()
        moderate_ans = AIcommunicator.moderate(content)
        if moderate_ans != '':
            return f'Error: your input violate OpenAI moderation rule: {moderate_ans}'
        completion = None
        if AIcommunicator.chat_model in ['gpt-3.5-turbo']:
            completion = openai.ChatCompletion.create(model=AIcommunicator.chat_model, messages=[{"role": role, "content": content}])
        else:
            completion = openai.Completion.create(model=AIcommunicator.chat_model, prompt=content)
        ### steram=True will not output usage !?   # ref: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
#        for chunk in completion:
#            print(chunk)
        if cls.should_show_elapsed_time:
            check_elapsed_time(start_time)   # 1.19s
#        print(completion['usage'])
        est_cost = 0
		if 'usage' in completion:
			num_prompt_tokens = completion['usage']['prompt_tokens']
			num_completion_tokens = completion['usage']['completion_tokens']
			est_cost_model = AIcommunicator.chat_model
			
			if 'gpt-3.5-turbo' in AIcommunicator.chat_model:
				est_cost = (num_prompt_tokens * 2 + num_completion_tokens) * 0.002/1000
				est_cost_model = 'gpt-3.5-turbo'
			elif 'gpt-4' in AIcommunicator.chat_model:
				est_cost = num_prompt_tokens * 0.06/1000 + num_completion_tokens * 0.12/1000
				est_cost_model = 'gpt-4'
			elif 'ada' in AIcommunicator.chat_model:
				est_cost = (num_prompt_tokens * 2 + num_completion_tokens) * 0.0016/1000
				est_cost_model = 'ada'
			elif 'babbage' in AIcommunicator.chat_model:
				est_cost = (num_prompt_tokens * 2 + num_completion_tokens) * 0.0024/1000
				est_cost_model = 'babbage'
			elif 'curie' in AIcommunicator.chat_model:
				est_cost = (num_prompt_tokens + num_completion_tokens) * 0.0120/1000
				est_cost_model = 'curie'
			elif 'davinci' in AIcommunicator.chat_model:
				est_cost = (num_prompt_tokens + num_completion_tokens) * 0.1200/1000
				est_cost_model = 'davinci'
			else:
				_print(f'Warning: unknown pricing for model {AIcommunicator.chat_model}, so I just use gpt-3.5-turbo instead')
				est_cost = (num_prompt_tokens * 2 + num_completion_tokens) * 0.002/1000
				est_cost_model = 'gpt-3.5-turbo'
			est_cost = round(est_cost, 7)
			total_est_cost += est_cost
			total_est_cost = round(total_est_cost, 7)
			if est_cost_model+' prompt' not in total_cost_items:
				total_cost_items[est_cost_model+' prompt'] = 0
			total_cost_items[est_cost_model+' prompt'] += num_prompt_tokens
			if est_cost_model+' completion' not in total_cost_items:
				total_cost_items[est_cost_model+' completion'] = 0
			total_cost_items[est_cost_model+' completion'] += num_completion_tokens
			update_total_est_cost_file(total_est_cost, program_datetime_stamp, total_cost_items)
			if cls.should_show_cost:
				_print(f'Estimated cost ${np.format_float_positional(est_cost)} for {num_prompt_tokens} prompt tokens + {num_completion_tokens} completion tokens with model {est_cost_model}')
		else:
			_print(f'Cannot estimate cost with no numbers of tokens returned')
        if AIcommunicator.chat_model in ['gpt-3.5-turbo']:
            return completion.choices[0].message.content
        else:
            return completion.choices[0].text

    @classmethod
    def generate_image(cls, prompt, n=1):
        global total_est_cost
        est_cost = 0
        if cls.image_size == 1024:
            est_cost = 0.020 * n
        elif cls.image_size == 512:
            est_cost = 0.018 * n
        elif cls.image_size == 256:
            est_cost = 0.016 * n
        else:
            raise ValueError('AIcommunicator.image_size should be one of 1024, 512 or 256')
        est_cost = round(est_cost, 3)
        total_est_cost += est_cost
        total_est_cost = round(total_est_cost, 7)
        if f'{cls.image_size}x{cls.image_size} image' not in total_cost_items:
            total_cost_items[f'{cls.image_size}x{cls.image_size} image'] = 0
        total_cost_items[f'{cls.image_size}x{cls.image_size} image'] += n
        update_total_est_cost_file(total_est_cost, program_datetime_stamp, total_cost_items)
        if cls.should_show_cost:
            _print(f'Estimated cost ${np.format_float_positional(est_cost)} for {n} {cls.image_size}x{cls.image_size} image')
        start_time = time.time()
        image_resp = openai.Image.create(prompt=prompt, n=n, size=f'{cls.image_size}x{cls.image_size}')
        if cls.should_show_elapsed_time:
            check_elapsed_time(start_time)   # 4.95s
        return [x['url'] for x in image_resp['data']]

    @classmethod
    def list_models(cls):
        if len(cls.OpenAI_models) == 0:
            models = openai.Model.list()
            for one_model in models.data:
                cls.OpenAI_models.append(one_model.id)
        return cls.OpenAI_models


def download_image(image_url, output_image_filename):
    img_data = requests.get(image_url).content
    with open(output_image_filename, 'wb') as handler:
        handler.write(img_data)


def print_help():
    usage = '''
Type  'exit' and Enter  or  Ctrl-c  to end program
Type  'Setting' and Enter  to configure settings
Type  'L' and Enter  to start recording microphone
    (automatically stop after 1 sec of silence)
Type or say  'Please draw picture of ...' or 'Can you draw picture of ...' or '请画图...'  to ask DALL-E draw a image
Type or say  'Please draw picture again' or '请重画'  to ask DALL-E draw image with previous prompt again
'''
    print_formatted_text(FormattedText([('orange', usage)]))

draw_picture_pattern_1 = re.compile('Please draw [A-Za-z0-9 ]* pictures* of (.*)')
draw_picture_pattern_2 = re.compile('Can you draw [A-Za-z0-9 ]* pictures* of (.*)')
draw_picture_pattern_3 = re.compile('请画图(.*)')
empty_pattern = re.compile('^[ \n\r\t]*$')
draw_picture_again_pattern_1 = re.compile('Please draw [A-Za-z0-9 ]* pictures again')
draw_picture_again_pattern_2 = re.compile('请重画')

def main():
#    recorder = Recorder()
    recording_filename_prefix = 'recording'
    previous_image_prompt = None
    image_filename_prefix = 'image'
    
    try:
        suggested_dBFS_threshold, avg_dBFS_vec = Recorder.suggest_dBFS_threshold()
        should_update_dBFS_threshold = prompt(f'Should I update Recorder.silence_dBFS_threshold to {suggested_dBFS_threshold}? ', default='Y')
        should_update_dBFS_threshold = strtobool(should_update_dBFS_threshold, False)
        if should_update_dBFS_threshold:
            setattr(Recorder, 'silence_dBFS_threshold', suggested_dBFS_threshold)
    except KeyboardInterrupt:
        pass
    
    session = PromptSession()
    print_help()

    while True:
        try:
            user_input = session.prompt('> ', multiline=Setting.prompt_allow_multiline)
            if user_input == "exit":
                break
            if user_input == "Setting":
                Setting.show_setting()
                user_input = ''
                continue
            if user_input == 'L':
                try:
                    recording_filename = recording_filename_prefix + '.' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.wav'
                    print_formatted_text(FormattedText([('yellow', f'Now recording ...')]))
                    Recorder.record_to(recording_filename)
                    is_correct_ans = prompt('Do you think the recording is complete? ', default='Y')
                    is_recording_correct = strtobool(is_correct_ans, False)
                    while not is_recording_correct:
                        os.remove(recording_filename)
                        recording_filename = recording_filename_prefix + '.' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.wav'
                        print_formatted_text(FormattedText([('yellow', f'Now recording ...')]))
                        Recorder.record_to(recording_filename)
                        is_correct_ans = prompt('Do you think the recording is complete? ', default='Y')
                        is_recording_correct = strtobool(is_correct_ans, False)
                    
                    print_formatted_text(FormattedText([('violet', f'Now transcripting by OpenAI-whisper ...')]))
                    transcript = AIcommunicator.recognize_speech_whisper(recording_filename)
                    print_formatted_text(FormattedText([('yellow', f'Recognized: {transcript}')]))
                    is_correct_ans = prompt('Do you think the transcription is accurate? ', default='Y')
                    is_transcript_correct = strtobool(is_correct_ans, False)
                    while not is_transcript_correct:
                        print_formatted_text(FormattedText([('violet', f'Now transcripting by OpenAI-whisper ...')]))
                        transcript = AIcommunicator.recognize_speech_whisper(recording_filename)
                        print_formatted_text(FormattedText([('yellow', f'Recognized: {transcript}')]))
                        is_correct_ans = prompt('Do you think the transcription is accurate? ', default='Y')
                        is_transcript_correct = strtobool(is_correct_ans, False)
                    user_input = transcript
                    os.remove(recording_filename)
                    if empty_pattern.search(user_input):
                        continue
                except KeyboardInterrupt:
                    # Ctrl+C, exit loop
                    user_input = ''
                except Exception as e:
                    _print(f'{e}')
                    user_input = ''
            
            image_prompt = None
            search_rlt = draw_picture_pattern_1.search(user_input)
            if search_rlt is not None:
                image_prompt = search_rlt.group(1)
                if empty_pattern.search(image_prompt) is not None:
                    image_prompt = None
            search_rlt = draw_picture_pattern_2.search(user_input)
            if search_rlt is not None:
                image_prompt = search_rlt.group(1)
                if empty_pattern.search(image_prompt) is not None:
                    image_prompt = None
            search_rlt = draw_picture_pattern_3.search(user_input)
            if search_rlt is not None:
                image_prompt = search_rlt.group(1)
                if empty_pattern.search(image_prompt) is not None:
                    image_prompt = None
            
            search_rlt = draw_picture_again_pattern_1.search(user_input)
            if search_rlt is not None:
                image_prompt = previous_image_prompt
            search_rlt = draw_picture_again_pattern_2.search(user_input)
            if search_rlt is not None:
                image_prompt = previous_image_prompt
            
            if image_prompt is not None:
                previous_image_prompt = image_prompt
                print_formatted_text(FormattedText([('cyan', f'Now asking DALL-E to draw: {image_prompt}')]))
                image_urls = AIcommunicator.generate_image(image_prompt)
                for image_url in image_urls:
                    image_filename = image_filename_prefix + '.' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.jpg'
                    download_image(image_url, image_filename)
                    print_formatted_text(FormattedText([('cyan', f'Download generated image to {image_filename}')]))
            else:
                if empty_pattern.search(user_input):
                    continue
                print_formatted_text(FormattedText([('violet', f'Now waiting for ChatGPT response ...')]))
                ans = AIcommunicator.chat(user_input)
                if ans.startswith('Error:'):
                    print_formatted_text(FormattedText([('red', ans)]))
                else:
                    print_formatted_text(FormattedText([('lightgreen', ans)]))
                if AIcommunicator.should_speak:
                    print_formatted_text(FormattedText([('violet', f'Now asking gtts to speak ...')]))
                    AIcommunicator.gtts_speak(ans)
        except KeyboardInterrupt:
            # Ctrl+C, exit loop
            break

if __name__ == "__main__":
    main()

