# 1. Usar um atalho para gravar minha voz
# 2. Transcrever o audio para texto (em portugues) -> Whisper
# 3. De posse deste texto, quero jogar em uma LLM -> Agente
# 4. De possa da resposta da LLM, quero utilizar um modelo de TTS (API da OpenAI)
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
from pynput import keyboard
import sounddevice as sd
import wave
import os
import numpy as np


load_dotenv(find_dotenv())
client = openai.Client()

class TalkingLLM():
  def __init__(self):
    self.is_recording = False
    self.audio_data = []
    self.samplerate = 44100
    self.channels = 1
    self.dtype = 'int16'
  
  def create_agent(self):
    agent_prompt_prefix = """
    Você se chama Tommy e está trabalhando com dataframe pandas no Python. O nome do dataframe é df.
    """
    df = pd.read_csv("../data/df_rent.csv")
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(model="gpt-4o-mini"),
        df,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        prefix=agent_prompt_prefix,
        verbose=True,
    )
  
  def start_stop_recording(self):
    if self.is_recording:
      self.is_recording = False
      self.save_and_transcribe()
      self.audio_data = []
    else:
      print("Starting record")
      self.audio_data = []
      self.is_recording = True
  
  def save_and_transcribe(self): 
    print("Saving the recording...")
    if "temp.wav" in os.listdir(): os.remove("temp.wav")
    wav_file = wave.open("test.wav", 'wb')
    wav_file.setnchannels(self.channels)
    wav_file.setsampwidth(2)  # Corrigido para usar a largura de amostra para int16 diretamente
    wav_file.setframerate(self.samplerate)
    wav_file.writeframes(np.array(self.audio_data, dtype=self.dtype))
    wav_file.close()
  
  def convert_and_play(self):
    pass
  
  def run(self):
    print('Estou rodando')
    
    def callback(indata, frame_count, time_info, status):
      if self.is_recording:
        self.audio_data.extend(indata.copy())
    
    with sd.InputStream(samplerate=self.samplerate, 
                            channels=self.channels, 
                            dtype=self.dtype , 
                            callback=callback):
      def on_activate():
        self.start_stop_recording()

      def for_canonical(f):
        return lambda k: f(l.canonical(k))

      hotkey = keyboard.HotKey(
          keyboard.HotKey.parse('<ctrl>'),
          on_activate)
      with keyboard.Listener(
              on_press=for_canonical(hotkey.press),
              on_release=for_canonical(hotkey.release)) as l:
        l.join()
    
  
  
if __name__ == "__main__":
  talking_llm = TalkingLLM()
  talking_llm.run()
  