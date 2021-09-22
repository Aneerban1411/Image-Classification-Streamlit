import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# Create application title and file uploader widget.
st.title("OpenCV Deep Learning based Image Classification")
img_file_buffer = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])


# Function for detecting facses in an image.
def classify(model, image, class_names):
    
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
    out_text = f"Class: {out_name}, Confidence: {final_prob:.3f}"
    return out_text




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

if img_file_buffer is not None:
    # Read the file and convert it to opencv Image.
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    # Loads image in a BGR channel order.
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    st.image(image[:,:,::-1])

    # Call the classification model to detect faces in the image.
    detections = classify(net, image, class_names)

    st.header(detections)
