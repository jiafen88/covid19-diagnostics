
# Core Pkgs
import streamlit as st
st.set_page_config(page_title="Covid-19 Detection Tool", page_icon="covid19.jpeg", layout='centered', initial_sidebar_state='auto')

import os
import time
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import tensorflow as tf

def main():
	"""Covid-19 Detection Program from Chest X-Ray"""
	html_templ = """
	<div style="background-color:blue;padding:10px;">
	<h1 style="color:yellow">Covid-19 Detection from Chest X-Ray Images</h1>
	</div>
	"""

	st.markdown(html_templ, unsafe_allow_html=True)
	st.write("A simple proposal for Covid-19 Diagnosis powered by Deep Learning and Streamlit")

	st.sidebar.image("covid19.jpeg",width=300)

	image_file = st.sidebar.file_uploader("Upload an X-Ray Image (jpg, png or jpeg)",type=['jpg','png','jpeg'])

	if image_file == None:
		image_file = "COVID-33.png"

	if image_file is not None:
		our_image = Image.open(image_file)

		if st.sidebar.button("Image Preview"):
			st.sidebar.image(our_image,width=300)

		activities = ["Image Enhancement","Diagnosis", "Disclaimer and Info"]
		choice = st.sidebar.selectbox("Select Activty", activities)

		if choice == 'Image Enhancement':
			st.subheader("Image Enhancement")

			enhance_type = st.sidebar.radio("Enhance Type", ["Original","Contrast","Brightness"])

			if enhance_type == 'Contrast':
				c_rate = st.slider("Contrast", 0.5, 5.0)
				enhancer = ImageEnhance.Contrast(our_image)
				img_output = enhancer.enhance(c_rate)
				st.image(img_output,use_column_width=True)

			elif enhance_type == 'Brightness':
				c_rate = st.slider("Brightness", 0.5, 5.0)
				enhancer = ImageEnhance.Brightness(our_image)
				img_output = enhancer.enhance(c_rate)
				st.image(img_output, width=600, use_column_width=True)

			else:
				st.text("Original Image")
				st.image(our_image,width=600,use_column_width=True)


		elif choice == 'Diagnosis':

			if st.sidebar.button("Diagnosis"):

				# Image to Black and White
				new_img = np.array(our_image.convert('RGB')) #our image is binary we have to convert it in array
				new_img = cv2.cvtColor(new_img, 1) # 0 is original, 1 is grayscale
				gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
				st.text("Chest X-Ray")
				st.image(gray, use_column_width=True)

				# PX-Ray (Image) Preprocessing - Covid19_CNN_Classifier
				# IMG_SIZE = (200,200)
				# img = cv2.equalizeHist(gray)
				# img = cv2.resize(img,IMG_SIZE)
				# img = img/255. #Normalization
				# X_Ray = img.reshape(1,200,200,1)

				# preprocess for cnn_model.h5
				z_img = cv2.resize(new_img, (70, 70)) / 255.0
				X_Ray = z_img.reshape(1, z_img.shape[0], z_img.shape[1], z_img.shape[2])

				# Pre-Trained CNN Model Importing
				# model = tf.keras.models.load_model("Covid19_CNN_Classifier.h5")
				model = tf.keras.models.load_model("cnn_model.h5")

				# Diagnosis (Prevision=Binary Classification)
				#diagnosis = model.predict(X_Ray)
				diagnosis_proba = model.predict(X_Ray)
				diagnosis = [1 if prob > 0.5 else 0 for prob in np.ravel(diagnosis_proba)]

				diagnosis_proba = np.ravel(diagnosis_proba)[0]
				diagnosis = diagnosis[0]

				probability_cov = diagnosis_proba * 100
				probability_no_cov = (1 - diagnosis_proba) * 100

				my_bar = st.sidebar.progress(0)

				for percent_complete in range(100):
					time.sleep(0.01)
					my_bar.progress(percent_complete + 1)

				# Diagnosis Cases: No-Covid=0, Covid=1
				if diagnosis == 0:
					st.sidebar.success("Result: NO COVID-19 (Probability: %.1f%%)" % (probability_no_cov))
				else:
					st.sidebar.error("Result: COVID-19 (Probability: %.1f%%)" % (probability_cov))
				st.warning("This Web App is just a DEMO only and the diagnosis has no clinical value.")

		else:
			st.subheader("Disclaimer and Info")
			st.subheader("Disclaimer")
			st.write("**This Web App is just a DEMO only and the diagnosis has no clinical value.**")
			st.subheader("Info")
			st.write("This Tool gets inspiration from the following works:")
			st.write("- [Detecting COVID-19 in X-ray images with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/)")
			st.write("- [Fighting Corona Virus with Artificial Intelligence & Deep Learning](https://www.youtube.com/watch?v=_bDHOwASVS4)")
			st.write("- [Updated Covid patient images from Kaggle](https://www.kaggle.com/code/sana306/detection-of-covid-positive-cases-using-dl/data?select=COVID-19_Radiography_Dataset)")
			st.write("The model was built from 3616 X-Ray images of patients infected by Covid-19 and 10192 X-Ray images of healthy people to train a Convolutional Neural Network in order to make a classification of pictures referring to infected and not-infected people.")
			st.write("The result was quite good since we got 96.1% accuracy on the training set and 90.4% accuracy on the test set. Unfortunately in our test we got 67 cases of 'False Negative', patients classified as healthy that actually are infected by Covid-19. It's very easy to understand that these cases can be a huge issue.")

	if st.sidebar.button("About the Author"):
		st.sidebar.subheader("Covid-19 Detection Tool")

if __name__ == '__main__':
		main()