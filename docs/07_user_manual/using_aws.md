This document contains information on the usage of AWS for this project.

### Getting a running AWS instance

The first step is to have a running AWS EC2 instance with a GPU. Follow these steps to reach that stage:

- After making an AWS account, request for GPU-based resources. This link contains information on how to do that: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html
- After the quota increase limit gets accepted, the next step is to set up an EC2 instance with a GPU. Here is a list of available GPU instance types: https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html
- For ease, an existing deep learning template (AMI) can be used: https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/ or an instance can be set up from scratch and then required software and packages can be installed. A list of required packages will be provided in the Readme of the repository.

Once we have an AWS EC2 instance with a GPU set up, we are good to go for the next step of model training.

### Running instance and training model

- Make sure the instance is running by right clicking on the instance and selecting `Instance State -> Start`.
- Then click on connect to get the command to ssh into the instance. If you do not already have a key-pair set up, you will need to create one and download it to your local machine in order to access the instance from terminal.
- The command to ssh into the instance should be run from the terminal after navigating to the folder where the key is present (or alternatively, providing the full path to the key in the command) which looks something like this:
  ```bash
  ssh -i "qio_key.pem" ubuntu@ec2-34-221-205-237.us-west-2.compute.amazonaws.com
  ```
- After logging into the machine, install the required software if the instance was created from a blank template or else, just move in the code from the repository the `ironn` directory to the instance. There are multiple ways to do this:
  - The first and the simplest way is to use the linux scp commands. Here is a great resource that provides all the ways the scp commands can be used to transfer files from and to a remote machine: https://unix.stackexchange.com/questions/188285/how-to-copy-a-file-from-a-remote-server-to-a-local-machine. For instance, if I have to transfer a file code.py from my local to the AWS instance, I would use something like:
    ```bash
    scp path_to_code_file/code.py ubuntu@ec2-34-221-205-237.us-west-2.compute.amazonaws.com:/home/ubuntu/dest_path_to_code/code.py
    ```
  - The second way would be to run a remote Jupyter notebook server to directly download and upload files. To set up such a connection, use this link: https://ljvmiranda921.github.io/notebook/2018/01/31/running-a-jupyter-notebook/
- After transferring the code, the next step is to transfer the data created using `export_poly_train_data.py`. All we need to transfer are the `training`, `labels` and the `testing` folders. The testing folders can be used to put in new files for prediction.
- The model can then be trained or just used for the inference using the steps mentioned in the `model_training.md` file.
- It would be a good idea to use `tmux` in order to run the model training process remotely so that connection losses do not interrupt the training process. Tmux should already come installed if using a template for EC2, otherise it can be installed easily on an Ubuntu Debian machine using:
    ```bash
    sudo apt-get update
    sudo apt-get install tmux
    ```
  Once installed, simply invoke a new session using:
    ```bash
    tmux new -s session_name
    ```
  And then run any commands for training or inference. If you want to close your local machine or want the process to run in the background, simply detach the tmux session using `Ctrl + b` (to invoke tmux commands) and then pressing `d` (for detach).
  If the session was already created earlier and the instance was not stopped, the session can be attached again using:
    ```bash
    tmux attach -t session_name
    ```
  If the instance was stopped, then a new session will have to be created again for the process.
  For a more comprehensive list of tmux commands, use this link: https://gist.github.com/MohamedAlaa/2961058
- The inference results will be present in the output directory specified while running the command for training or inference. This directory can be zipped and transferred again using the scp commands or the notebook interface to the local machine to look at the results.
- Once done, **do not forget to stop the instance using `Instance State -> Stop` from the right click menu of the instance in the AWS EC2 dashboard**.
