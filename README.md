1. Description:
The repository contains the code for a project focused on developing 3D uncertainty-aware deep learning models for pediatric brain tumor segmentation. It utilizes the BraTS-PEDs dataset and compares the performance of two primary architectures:
    3D U-net: Implemented in TensorFlow/Keras
    Swin-UNETR: Implemented in PyTorch
The pipeline handles 3D medical image preprocessing, model trainin,g and evaluation using metrics like Dice Coefficient and Hausdorff Distance.

2. Repository Structure
load_dataset.py: Handles data loading, normalization, 3D patch extraction, and data augmentation.
model_unet.py & model_swin_unetr.py: Define the neural network architectures.
metric_tf.py & metric_torch.py: Contain custom loss functions (BCE + Soft Dice) and evaluation metrics, including MC Dropout uncertainty mapping.
02_train_unet.ipynb / 02_train_unet.py: Training scripts for the TensorFlow 3D U-Net model.
02_train_swin.ipynb / 02_train_swin.py: Training scripts for the PyTorch Swin-UNETR model.
03_evaluate_models.ipynb: Testing and visualization notebook to evaluate models on unseen data and calculate MC dropout variance.
config.json & hparams.json: Configuration files for dataset paths and model hyperparameters.

3. Setup & Installation
To install dependencies, run:
    pip install -r requirements.txt

We highly recommend using a virtual environment. Detailed explanation are in setup.md

4. Execution
    
    A. Data & Configuration
        Open config.json and set "dataset_path" to the absolute file path of your BraTS-PEDs directory.
        Open hparams.json to adjust hyperparameters (e.g., patch size, batch size, learning rate) for both the 3D U-net and Swin-UNETR models.
        
    B. Train the Models
        You can execute the training pipeline via the provided Jupyter Notebooks or directly through the terminal using the .py files. To train via the terminal, run:
            python 02_train_swin.py
            python 02_train_unet.py
        Note: To run a quick test on a subset of the dataset, you can specify n_train and n_val within the dataset building sections of the training scripts (e.g., n_train=5, n_val=3). 
    
    C. Evaluate and Visualize
        Once training is complete, execute the 03_evaluate_models.ipynb notebook to test the models on unseen data. Running all cells in this notebook will:
        1. Load the trained model checkpoints.
        2. Run inference on the test dataset.
        3. Output the Loss, Dice Coefficient, and Hausdorff Distance for both architectures.
        4. Compute the MC dropout variance to estimate prediction uncertainty