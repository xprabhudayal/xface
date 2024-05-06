from deepface import DeepFace
model = DeepFace.build_model("Emotion")

import telegram
from telegram import Update
from telegram.ext import CommandHandler, Updater, MessageHandler, Filters
from google.colab import userdata
###
import io
import requests
from PIL import Image
import cv2  # Assuming OpenCV is installed (`pip install opencv-python`)
import numpy as np
from deepface import DeepFace
###

#this is used for getting the telegram token and would be a secret for everyone
#paste your telegram bot token here like

#TOKEN = "Your specified token"
TOKEN = userdata.get('XFACE')


def start(update, context):
    return update.message.reply_text("Hi, this is XFace bot. I can analyze a image, and provide 'Dominant Expression'")


#used to download the url image
def remote_image(url):
  response = requests.get(url, stream=True)
  response.raise_for_status()  # Raise an exception for non-200 status codes
  img = Image.open(io.BytesIO(response.content))
  return img



def analyze_image(url):
  try:
    final = cv2.cvtColor(np.array(url), cv2.COLOR_BGR2RGB)
    output = DeepFace.analyze(final, actions=['emotion'])

    # Optionally display the analyzed image with annotations
    if output[0]['dominant_emotion']:
      return f"{(output[0]['face_confidence'])*100}% {output[0]['dominant_emotion']}"
  except:
    return "The Face can't be detected, try cropping only the face part."


def msg_handler(url : str):
  if "https" or "http" in url:
    print("url based detection")
    return analyze_image(url)
  else:
    return "document based detection"


def main(update : Update, context):
  if update.message.text :
    url = update.message.text
    result = analyze_image(remote_image(url))
    update.message.reply_text(result)

  else:
    try:
      file_id = update.message.document.file_id
      file_info = context.bot.get_file(file_id)
      file_url = f'{file_info.file_path}'

      #download the image locally to analyze
      print(file_url)
      img = remote_image(file_url)
      result = analyze_image(img)

      return update.message.reply_text(result)

    except:
      return update.message.reply_text("Error, Uncheck the 'Compress' option and try again!")




# Create the Updater and pass in your bot's token
updater = Updater(TOKEN, use_context=True)

# Get the dispatcher to register handlers
dispatcher = updater.dispatcher

# Add command handlers
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.text, main))
dispatcher.add_handler(MessageHandler(Filters.photo, main))
dispatcher.add_handler(MessageHandler(Filters.document, main))

# Start the Bot
updater.start_polling()
updater.idle()
