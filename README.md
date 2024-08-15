1. Overview

   This is the code for training and evaluating VQA model on Malayalam version of VQA-X dataset. The malayalam version of the dataset can be found at [huggingface dataset hub](https://huggingface.co/datasets/bipin/ml_vqa).

3. Structure of the code
   * data.py - contain the code to load and pre-process the data used for training the VQA model model.
   * model.py - contain the model architecture and the trainer class.
   * pre_train.py - the code to run the first phase (pre-training) of the model training.
   * finetune.py - code to run the second phase (finetuning) of the model training.
   * test.py - code to evaluate the trained model on the test set of VQA-X dataset.
   * utils.py - helper functions and classes for training and evaluating the model.
   * demo.py - script to run the demo of the project. You will get an interface to upload an image and a question, the model generated answer will be shown in the same interface.

3. Training the model

   The model can be trained by following the below steps:
   
     i. Install the required packages by running the following command on your terminal:

      ```
       pip install -r requirements.txt
      ```
   
     ii. Run the pre-training phase of the model by running the following command (model will be saved as `VQAModel-step1.pt`):

      ```
       python pre_train.py
      ```

     iii. Run the finetuning phase of the model using the following command (model will be saved as `VQAModel-step2.pt`):

      ```
       python finetune.py
      ```
    
5. Evaluating the model on VQA-X test set:

   To run the evaluation run the below command. The script will look for saved weights after the finetuning phase in the local directory
   and load it. If it doesn't find `VQAModel-step2.pt` file locally, it will automatically download the weights stored in huggingface hub:

      ```
      python test.py
      ```

   The results of the evaluation will be displayed on the terminal.
   
6. Running the demo

   The demo is created using the [Gradio library](https://www.gradio.app/). You can run the following command in the terminal to access the demo:
   
     ```
     python demo.py
     ```
   
   Running the above command will give you a public as well as a local URL to access the demo. The public URL is valid for 72 hours and can be shared with others. The local URL can only be accessed from your local system.