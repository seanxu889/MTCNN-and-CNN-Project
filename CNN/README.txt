[CS 613 - Final Project CNN Source Code]

Name: Shaoshu Xu

=================================================================

Environment:

1. tensorflow 1.3 && python3.6: [https://github.com/tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)
2. numpy [‘pip install numpy’]
3. PIL [‘pip install Pillow’]
4. matplotlib [‘python -m pip install matplotlib’]

=================================================================

Prepare Data:

Notice: You should be at ‘ROOT_DIR/‘ if you want to run the following command.

1. The JAFFE dataset is in ‘./Project’.

2. The resized face-images from MTCNN output should be ‘./mtcnn_output/resizedFace’, with the size 32*32 RGB scale.

=================================================================

Training and Testing Example:

1. Run ‘cnn_classifier.py’ to train the model. It will print the accuracy and time-passed of each epoch. After training process end, it will output the prediction results of the face images in the ./mtcnn_output/resizedFace folder. 

2. After training, you can reload the testing images into ‘val_img’ using the code at line 121 to 135. And then, running the command ’print(prediction_label(val_img))’ you will get the new prediction results.

=================================================================

Additional: If you have any questions about the files or compile problem, please feel free to email me. [sx66@drexel.edu]

Thank you!
