#DL_Mini_Project.py 
import streamlit as st
import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import base64


def set_bg_hack(main_bg):
        '''
        A function to unpack an image from root folder and set as bg.
        The bg will be static and won't take resolution of device into account.
        Returns
        -------
        The background.
        '''
        # set bg name
        main_bg_ext = "jpg"
            
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
set_bg_hack('background.jpg')

def load_model():
  model1=tf.keras.models.load_model('vgg16/saved_model.pb')
  model2=tf.keras.models.load_model('resnet50/saved_model.pb')
  return model1,model2
with st.spinner('Models are being loaded..'):
  model1,model2=load_model()

def import_and_predict(image_data, model):
  size = (256,256)    
  image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
  image = np.asarray(image)
  img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  img_resize = (cv2.resize(img, dsize=(256, 256),interpolation=cv2.INTER_CUBIC))/255.
  img_reshape = img_resize[np.newaxis,...]
  prediction = model.predict(img_reshape)
  return prediction

CATEGORIES = ['Covid', 'Normal', 'Viral Pneumonia']
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Covid-19 Detection using Deep Learning')
st.text('Upload the chest X-Ray image to classify')
uploaded_file = st.file_uploader("Choose an Image....",type = ['png','jpeg','jpg'])
if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img,caption = "Image Uploaded")

if st.button('PREDICT'):
  st.write('Result....')
  vgg16_predictions = import_and_predict(img, model1)
  resnet50_predictions = import_and_predict(img, model2)

  score1 = tf.nn.softmax(vgg16_predictions[0])
  score2 = tf.nn.softmax(resnet50_predictions[0])
  score1 = score1.numpy()
  score2 = score2.numpy()
  vgg16_scores,resnet50_scores = [],[]
  st.title(f'vgg16 predicted :{CATEGORIES[np.argmax(score1)]}')
  st.title(f'resnet50 predicted :{CATEGORIES[np.argmax(score2)]}')
  print(f'vgg16 output:{CATEGORIES[np.argmax(score1)]}')
  print(f'resnet50 output:{CATEGORIES[np.argmax(score2)]}')
  st.title('vgg16 scores :')
  for index,item in enumerate(CATEGORIES):                  #prints the probability of all the categories
    vgg16_scores.append(score1[index]*100)
    st.write(f'{item} : {vgg16_scores[index]}%')
    print(f'{item} : {vgg16_scores[index]}%')
  st.bar_chart(data = vgg16_scores, width=0, height=0, use_container_width=True)

  st.title('resnet50 scores :')
  for index,item in enumerate(CATEGORIES):                  #prints the probability of all the categories
    resnet50_scores.append(score2[index])
    st.write(f'{item} : {resnet50_scores[index]*100}%')
    print(f'{item} : {resnet50_scores[index]*100}%')
  st.bar_chart(data = resnet50_scores, width=0, height=0, use_container_width=True)

  for i in range(0,1):
    #if scores[2] > scores[0] or scores[1]:
     # st.title("You may or may not have Viral Pneumonia, kindly consult an expert!!")
     # break
    if (CATEGORIES[np.argmax(score1)] == "Covid" and CATEGORIES[np.argmax(score2)] == "Covid") and ((vgg16_scores[0] < vgg16_scores[1] + vgg16_scores[2]) or (resnet50_scores[0] < resnet50_scores[1] + resnet50_scores[2])):
      st.title("Risk of Covid-19 and a trace of Viral Pneumonia found, extreme care is advised!!")
      break
    elif (CATEGORIES[np.argmax(score1)] == "Covid" and CATEGORIES[np.argmax(score2)] == "Covid"):
      st.title("Covid-19 Affirmative, extreme care is advised!!")
      break
    elif CATEGORIES[np.argmax(score1)] == "Normal" and CATEGORIES[np.argmax(score2)] == "Normal":
      st.title("Normal, no additional diagnosis is required!!")
      break 
    elif CATEGORIES[np.argmax(score1)] == "Viral Pneumonia" and CATEGORIES[np.argmax(score2)] == "Viral Pneumonia":
      st.title("Viral Pneumonia Affirmative, extreme care is advised!!")
      break 
    elif (CATEGORIES[np.argmax(score1)] == "Covid" and CATEGORIES[np.argmax(score2)] == "Normal") or (CATEGORIES[np.argmax(score2)] == "Covid" and CATEGORIES[np.argmax(score1)] == "Normal"):
      st.title("You may or may not have Covid-19, additional diagnosis is required!!")
      break
    elif (CATEGORIES[np.argmax(score1)] == "Covid" and CATEGORIES[np.argmax(score2)] == "Viral Pneumonia") or (CATEGORIES[np.argmax(score2)] == "Covid" and CATEGORIES[np.argmax(score1)] == "Viral Pneumonia"):
      st.title("Complication of Covid and Pneumonia are detected, additional diagnosis with an extreme care is advised")
      break
    elif (CATEGORIES[np.argmax(score1)] == "Normal" and CATEGORIES[np.argmax(score2)] == "Viral Pneumonia") or (CATEGORIES[np.argmax(score2)] == "Normal" and CATEGORIES[np.argmax(score1)] == "Viral Pneumonia"):
      st.title("Risk of Viral Pneumonia, additional diagnosis is required!!")
      break 
