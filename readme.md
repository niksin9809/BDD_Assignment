# BDD Dataset Analysis and Object Detection
The repository contains the code files for Exploratory Data Analysis and training a YOLO model for object detection.

## How to use the repository
- Using Dockerfile
    - Clone the repository and open the directory
    - Copy the dataset to the Assets/Dataset folder 
    - Build a docker image using the command (docker build -t nikhil_assignment:1 .)
    - Run the docker container using the command (docker run --gpus all -it -v {path to current folder}:/app nikhil_assignment:1) . Allow access to gpus and mount the current directory.
    - Open VS Code or any other IDE of preference
    - Connect the IDE to the running container
    - Open the working directory and run the required files
- Using Google Colab
    - Open Google colab
    - Clone the repository and install the requirements.txt using the command ()
    - Upload the dataset to colab or Load it from Google Drive by mounting the drive to your colab instance
    - Put the dataset in the Assets/Dataset folder
    - Open and run the desired files in the repository

**For training, the data was converted to YOLO training format and the codes to do the same are present in the repository**