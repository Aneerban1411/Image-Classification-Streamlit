from requests.models import MissingSchema
import streamlit as st
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO


# Create application title and file uploader widget.
st.title("OpenCV Deep Learning based Image Classification")
mode = st.selectbox('Upload image from Disc or Web?', ('None', 'Disc', 'Web'))


# Function for detecting facses in an image.
def classify(model, image, class_names):

    # Converting RGB image to BGR OpenCV image
    image = np.uint8(image)[:,:,::-1]
    
    # create blob from image
    blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.01, size=(224, 224), 
                                mean=(104, 117, 123))
    # set the input blob for the neural network
    model.setInput(blob)
    # forward pass image blog through the model
    outputs = model.forward()

    final_outputs = outputs[0]
    # make all the outputs 1D
    final_outputs = final_outputs.reshape(1000, 1)
    # get the class label
    label_id = np.argmax(final_outputs)
    # convert the output scores to softmax probabilities
    probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
    # get the final highest probability
    final_prob = np.max(probs) * 100.
    # map the max confidence to the class label names
    out_name = class_names[label_id]
    out_text = f"Class: {out_name}, Confidence: {final_prob:.1f}%"
    return out_text

def header(text):
    st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;" align="center">{text}</p>', unsafe_allow_html=True)





# Function to load the DNN model.
@st.cache(allow_output_mutation=True)
def load_model():
    # read the ImageNet class names
    with open('classification_classes_ILSVRC2012.txt', 'r') as f:
        image_net_names = f.read().split('\n')
    # final class names (just the first word of the many ImageNet names for one image)
    class_names = [name.split(',')[0] for name in image_net_names]

    # load the neural network model
    model = cv2.dnn.readNet(model='DenseNet_121.caffemodel', 
                        config='DenseNet_121.prototxt', 
                        framework='Caffe')
    return model, class_names

net, class_names = load_model()

if mode == 'Disc':
    img_file_buffer = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])
    if img_file_buffer is not None:
        # Read the file and convert it to opencv Image.
        image = np.array(Image.open(img_file_buffer))
        st.image(image)

        # Call the classification model to detect faces in the image.
        detections = classify(net, image, class_names)

        header(detections)
elif mode == 'Web':
    url = st.text_input('Enter URL')

    if url != '':
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            st.image(image)

            # Call the classification model to detect faces in the image.
            detections = classify(net, image, class_names)

            header(detections)
        except MissingSchema:
            st.header('Invalid URL, Try Again!')
        except UnidentifiedImageError:
            st.header('URL has no Image, Try Again!')
