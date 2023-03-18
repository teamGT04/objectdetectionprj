import pyttsx3
speech=pyttsx3.init()
ans=input("what you want to speak")
speech.say(ans)
speech.runAndWait()