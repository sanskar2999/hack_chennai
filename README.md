# tf-pose-estimation

'Openpose' for human pose estimation have been implemented using Tensorflow. It also provides several variants that have made some changes to the network structure for **real-time processing on the CPU or low-power embedded devices.**
Original Repo(Openpose Tensorflow) : https://github.com/ildoonet/tf-pose-estimation
### Dependencies

You need dependencies below.

- python3
- tensorflow 1.4.1+
- OpenCV, protobuf, python3-tk
- Keras
-Scikitlearn

## Models


- cmu 
  - the model based VGG pretrained network which described in the original paper.
    Download Weights:
    http://www.mediafire.com/file/1pyjsjl0p93x27c/graph_freeze.pb 
    http://www.mediafire.com/file/qlzzr20mpocnpa3/graph_opt.pb 
    http://www.mediafire.com/file/i72ll9k5i7x6qfh/graph.pb 

- Mobilenet_thin
    
