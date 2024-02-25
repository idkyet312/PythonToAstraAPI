from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from langchain.memory import CassandraChatMessageHistory, ConversationBufferMemory
from langchain_openai import OpenAI
#from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
#from langchain_community.llms import BaseLLM
#from gtts import gTTS
import os
#import speech_recognition as sr
#import pygame

import time

#import simpleaudio as sa
import threading

import logging

# Configure the root logger directly
logging.getLogger().setLevel(logging.ERROR)

#import cassio
from flask import Flask

import os
from dotenv import load_dotenv
load_dotenv(".env")


app = Flask(__name__)






# Initialize recognizer
#recognizer = sr.Recognizer()


# Use the microphone as the source for input

def quit():
    pygame.quit()
    sys.exit()




'''
def record_speech():
    using = False
    with sr.Microphone() as source:
        #print(source)
        print("\nPlease say something...")

        # Listen for the first phrase and extract it into audio data
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source)
        print("...\n")
        try:
            # Recognize speech using Google Web Speech API
            text = recognizer.recognize_google(audio_data)
            print("You said: " + text + "\n")
        except sr.UnknownValueError:
            # API was unable to understand the audio
            print("Google Web Speech API could not understand the audio.")
            record_speech()
        except sr.RequestError as e:
            # API was unreachable or unresponsive
            print(f"Could not request results from Google Web Speech API; {e}")
    try:
        if using == False:
            using = True
            response = llm_chain.predict(human_input=text)
            print(response.strip())
            if "Game over." in str(response.strip()):
                print("/nYOU ARE DEAD!!!")
                quit()
    except:
        print("ERROR")
'''

'''def stop_speech():
    sound.stop()

def play_speech(text, stop_after=8):
    global stop_threads
    pygame.mixer.init()

    # Creating a temporary file for the speech
    #temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')

    filename = "D:\tmp\test.mp3"

    file = open("test.mp3")

    tts = gTTS(text=text, lang='en')
    tts.save(file.name)
    file.close()  # Explicitly close the file to ensure it's written

    #time.sleep(3)


    file_path = 'test.mp3'
    y, sr = librosa.load(file_path, sr=None)
    y_fast = librosa.effects.time_stretch(y, rate=2.7)  # Speeding up the audio

    # Convert the processed audio signal to pydub's AudioSegment
    y_fast_int16 = np.int16(y_fast / np.max(np.abs(y_fast)) * 32767)
    audio_segment = AudioSegment(
        y_fast_int16.tobytes(),
        frame_rate=sr,
        sample_width=y_fast_int16.dtype.itemsize,
        channels=1
    )

    # Convert AudioSegment to bytes, which simpleaudio can play back
    audio_bytes = audio_segment.raw_data

    # Play the audio bytes using simpleaudio
    play_obj = sa.play_buffer(audio_bytes, num_channels=1, bytes_per_sample=audio_segment.sample_width, sample_rate=audio_segment.frame_rate)

    # Wait for playback to finish before exiting
    play_obj.wait_done()'''

cloud_config= {
  'secure_connect_bundle': 'secure-connect-npc-one.zip'
}

'''with open("choose_your_own_adventure-token.json") as f:
    secrets = json.load(f)'''

#CLIENT_ID = secrets["clientId"]
#CLIENT_SECRET = secrets["secret"]
ASTRA_DB_KEYSPACE = "default_keyspace"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
USE_BETA = True
app = Flask(__name__)

cluster = Cluster(cloud=cloud_config, auth_provider=PlainTextAuthProvider("token", TOKEN), protocol_version = 4)

session = cluster.connect()

message_history = CassandraChatMessageHistory(
    session_id="anything",
    session=session,
    keyspace=ASTRA_DB_KEYSPACE,
    ttl_seconds=600
)

message_history.clear()

chat_history = ""
human_input = "start"

cass_buff_memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=message_history
)

template = """
SYSTEM

you are to simulate an AI of a user driven Adventure in 20 words that depends on the human_input to continue

A traveler named Elara seeks the lost Gem of Serenity. 
You must navigate her through challenges, choices, and consequences, 
dynamically adapting the tale based on the traveler's decisions. 
Your goal is to create a branching narrative experience where each choice 
leads to a new path or could lead to death, ultimately determining Elara's fate.
make dumb suggestions kill elara
DONT ASK to clarify if its a dumb choice

Here are some rules to follow:
1. Start by asking the player to choose some kind of weapons that will be used later in the game


Here is the chat history, use this to understand what to say next: {chat_history}

USER

{human_input}
AI:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template
)



llm = OpenAI(openai_api_key=OPENAI_API_KEY)
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=cass_buff_memory
)

choice = "start"

@app.route('/')
def hello_world():
    response = llm_chain.predict(human_input="bow")
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, port = port)

