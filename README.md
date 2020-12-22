# Overview

 Object detection is a sought-after field in computer vision with many applications. It is used from video surveillance, face detection to self-driving cars. We develop an application to interactively teach users how object detection is done using Convolutional neural networks. We have used a pre trained Coco ssd model for object detection, already available in tensoflow.js. Coco ssd uses mobilenet v2 architecture for feature extractions. Our application shows how these features are extracted to detect objects.

# Workflow

 The web app allows a user to upload any image and detect objects in it. We give a brief explanation of Convolution, activation function performed by the CNN for feature extraction. We also provide some interesting details useful in understanding the flow of the neural network. After having the knowledge of each of the terms, the user then gets to visually see how the layers of a convolution network look like by clicking on “Convolutional Neural Network for uploaded image” button. The display shows image outputs of each neuron in multiple layers.

----------

# Running the AI-for-kids project

- Download the code from GitHub
- Run the index.html file on your browser (recommended browser - chrome)
- No need to add any additional libraries or packages
- Start exploring!

# Below are the steps on how to use the application

1. The first page displayed gives the user an option to upload an image. After uploading the input image, click on Predict Image button to see what objects the model detects. The user can also click on model summary button to get additional details about the design of the network.

	![](https://raw.githubusercontent.com/KatnaturK/ai-for-kids/master/docs/starting_page.png)

2. After clicking the Predict Image button, a lot of information is displayed on the screen. Along with showing the output of the model, we give a brief overview about convolution, activation functions and other details to understand how the network predicts.

	![](https://raw.githubusercontent.com/KatnaturK/ai-for-kids/master/docs/prediction_panel.png)
	![](https://raw.githubusercontent.com/KatnaturK/ai-for-kids/master/docs/cnn-info-panel.png)

3. After learning about how the network using mathematical operations to learn, we next give the user an option to learn more about each layer outputs in the neural network. 

	![](https://raw.githubusercontent.com/KatnaturK/ai-for-kids/master/docs/cnn-flow-panel.png)

4. The multiple layer diagram shows the output of different convolution and activation layers. Click on any image in the layers to learn more about it.

	![](https://raw.githubusercontent.com/KatnaturK/ai-for-kids/master/docs/cnn-layers-panel.png)