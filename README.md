
egohands_dataset_clean.py - clean the data and turn into csv
generate_tfrecord.py - csv to tfrecord 
Label_map_util.py: The input is some label maps of images. The output are images with “Hand” labels. This script loads a label map, checks its validity, and then converts label map into “Hand” category. 
Visualization_util.py: The input is a testing image without any information. This script does not return a value. It only modifies the image itself and performs some visualization on the image. The image after modifying will have boxes which predict the position of hands. 
Eval.py: The input is test_tfrecord which are data about the true position of hands in test dataset. This script will compare the true boxes and predicted boxes to check the accuracy of our models.  
Export_inference_graph.py:  This script is a tool to export an object detection model for inference. This tool prepares an object detection tensorflow graph for inference using model configuration and an optional trained checkpoint. It outputs an inference graph, associated checkpoint files, a frozen inference graph and a model. 

Train.py - Take pipeline.config as input files.

project.ipynb -Loads the trained model and images to detect hands within the image

real_time_hand.py -Loads the trained model and use internal webcams to capture frames. Then detects if hands are within the frame.

project.html - html of project.ipynb 
