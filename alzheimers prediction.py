from tkinter import *
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk  # Added for image display
from tensorflow.keras.models import load_model

main = Tk()
main.title("Alzheimer's Prediction")
main.geometry("1300x1200")

mapping = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
filename = "my_dataset"
model = None

# Variables to store the uploaded image and prediction result
uploaded_image = None
prediction_result = StringVar()

def load():
    global model
    model = tf.keras.models.load_model('alzheimers_model.keras')
    model.summary()
    text.insert(END, "Alzheimer's Disease prediction model loaded..." + "\n")

def upload():
    text.delete('1.0', END)
    global filename
    filename = askopenfilename()
    text.insert(END, "File Uploaded: " + str(filename) + "\n")

def imagepreprocess():
    global img4, uploaded_image
    img3 = cv2.imread(filename)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    img3 = cv2.resize(img3, (224, 224))
    img3 = img3 / 255.0  # Normalize image values between 0 and 1
    img4 = np.reshape(img3, [1, 224, 224, 3])
    uploaded_image = Image.open(filename)
    uploaded_image = ImageTk.PhotoImage(uploaded_image)
    text.insert(END, "Image Preprocessing..." + "\n")

def show_prediction_result(image, prediction_name):
    # Create a new popup window for displaying the image and prediction result
    popup_window = Toplevel()
    popup_window.title("Prediction Result")
    popup_window.geometry("400x400")

    # Create a label to display the uploaded image
    image_label = Label(popup_window, image=image)
    image_label.pack()

    # Create a label to display the prediction result
    result_label = Label(popup_window, text="Predicted Output: " + prediction_name)
    result_label.pack()

def predict():
    global model, img4, uploaded_image, prediction_result
    if model is None:
        text.insert(END, "Please load the model first.\n")
        return

    disease_probs = model.predict(img4)
    prediction = np.argmax(disease_probs)
    prediction_name = mapping[prediction]
    text.insert(END, "Predicted output for uploaded Image: " + str(prediction_name) + "\n")

    # Display the uploaded image along with the prediction result in a popup window
    show_prediction_result(uploaded_image, prediction_name)

font = ('times', 16, 'bold')
title = Label(main, text="Alzheimer's Prediction From MRI Images")
title.config(bg='dark salmon', fg='black')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Image", command=upload)
upload.place(x=700, y=100)
upload.config(font=font1)

process = Button(main, text="Image Pre-Processing", command=imagepreprocess)
process.place(x=700, y=150)
process.config(font=font1)

ld = Button(main, text="Model Load", command=load)
ld.place(x=700, y=200)
ld.config(font=font1)

pred = Button(main, text="Predict", command=predict)
pred.place(x=700, y=250)
pred.config(font=font1)

font1 = ('times', 12, 'bold')
text = Text(main, height=30, width=80)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=100)
text.config(font=font1)

# Display the prediction result
result_label = Label(main, textvariable=prediction_result)
result_label.config(font=font1)
result_label.place(x=700, y=400)

main.config(bg='tan1')
main.mainloop()
