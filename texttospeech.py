import win32com.client as wincl
from time import sleep
import asyncio

def texttospeech(word):
    speaker_number = 1
    spk = wincl.Dispatch("SAPI.SpVoice")
    vcs = spk.GetVoices()
    SVSFlag = 11
    print(vcs.Item (speaker_number) .GetAttribute ("Name")) # speaker name
    spk.Voice
    spk.SetVoice(vcs.Item(speaker_number)) # set voice (see Windows Text-to-Speech settings)
    spk.Speak(word)
