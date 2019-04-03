[CS 613 - Final Project MTCNN Source Code]

Name: Shaoshu Xu

=================================================================

Environment:

1. tensorflow 1.3 && python3: [https://github.com/tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)
2. opencv 3.0 for python3.6 [‘pip install opencv-python’]
3. numpy 1.13 [‘pip install numpy’]

=================================================================

Prepare Data:

Notice: You should be at ‘ROOT_DIR/prepare_data/‘ if you want to run the following command.

1. Download Wider Face Training part only from [Official Website](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) and unzip to replace `WIDER_train`

2. Run ‘python gen_shuffle_data.py 12’ to generate 12net training data. Besides, ‘python gen_tfdata_12net.py’ provide you an example to build tfrecords file. Remember changing and adding necessary params.

3. Run ‘python tf_gen_12net_hard_example.py’ to generate hard sample. Run ‘python gen_shuffle_data.py 24’ to generate random cropped training data. Then run ‘python gen_tfdata_24net.py’ to combine these output and generate tfrecords file.

4. Similar to last step. Run ‘python gen_24net_hard_example.py’ to generate hard sample. Run ‘python gen_shuffle_data.py 48’ to generate random cropped training data. Then run ‘python gen_tfdata_48net.py’ to combine these output and generate tfrecords file.

=================================================================

Training Example:

Notice: You should be at ‘ROOT_DIR/‘ if you want to run the following command.

After finishing the step 2 above, you can run python src/mtcnn_pnet_test.py to do Pnet training. Similarly, after step 3 or step 4, you can run python src/mtcnn_rnet_test.py or python src/mtcnn_onet_test.py to train Rnet and Onet respectively.

The .txt files and plots in ’./src/loss’ folder show all the training process and recorded loss.

=================================================================

Testing Example:

Notice: You should be at ‘ROOT_DIR/‘ if you want to run the following command.

You can run this command line [‘python test_img.py test1.jpg --model_dir ./save_model/all_in_one/‘] in your terminal to test mtcnn with the saved model. 

The sample face image test1.jpg and the resized outputs are shown here.

=================================================================

Additional: If you have any questions about the files or compile problem, please feel free to email me. [sx66@drexel.edu]

Thank you!
