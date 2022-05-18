# covid19-diagnostics

## Disclaimer and Info
This Web App is just a DEMO of Deep learning and Streamlit deployment and the diagnosis has no clinical value.
This Tool gets inspiration from the following works:
- [Detecting COVID-19 in X-ray images with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/)
- [Fighting Corona Virus with Artificial Intelligence & Deep Learning](https://www.youtube.com/watch?v=_bDHOwASVS4)
- [Updated Covid patient images from Kaggle](https://www.kaggle.com/code/sana306/detection-of-covid-positive-cases-using-dl/data?select=COVID-19_Radiography_Dataset)

The model was originally trained from 3616 X-Ray images of patients infected by Covid-19 and 10192 X-Ray images of healthy people to train a Convolutional Neural Network in order to make a classification of pictures referring to infected and not-infected people.
The result was quite good since we got 96.1% accuracy on the training set and 90.4% accuracy on the test set.

Unfortunately in our test we got 67 cases of 'False Negative', patients classified as healthy that actually are infected by Covid-19. It's very easy to understand that these cases can be a huge issue.

You can check it out on [Heroku](https://covid19-positive-detection-cnn.herokuapp.com/).
