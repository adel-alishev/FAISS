import speech_recognition as sr

# obtain audio from the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)

# write audio to a RAW file
with open("audio/microphone-results.raw", "wb") as f:
    f.write(audio.get_raw_data())

# write audio to a WAV file
with open("audio/microphone-results.wav", "wb") as f:
    f.write(audio.get_wav_data())

# write audio to an AIFF file
with open("audio/microphone-results.aiff", "wb") as f:
    f.write(audio.get_aiff_data())

# write audio to a FLAC file
with open("audio/microphone-results.flac", "wb") as f:
    f.write(audio.get_flac_data())