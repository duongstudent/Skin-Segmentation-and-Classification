# README

## 0. Method

The proposed method follows a pipeline approach. The visual representation of the pipeline is shown below:

![Pipeline Method](https://github.com/duongstudent/Skin-Segmentation-and-Classification/raw/main/Results/pipeline_method.png)

Here are some examples illustrating the method:

![Some Examples](https://github.com/duongstudent/Skin-Segmentation-and-Classification/raw/main/Results/SomeExample.png)

## 1. Running the Demo Code

To run the demo code, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/duongstudent/Skin-Segmentation-and-Classification.git
   ```

2. Download the pre-trained model:
   - Download the model.zip file from [this link](https://drive.google.com/file/d/1lOhzuREewhoL9U9Cli9P6Z0cXTqdSVR2/view?usp=sharing)
   - Extract the contents and place them in the appropriate directory in the cloned repository.

3. Install Python 3.8.18

4. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the command:
   ```bash
   streamlit run app.py
   ```
   - Access the following URLs:
     - Local URL: [http://localhost:8501](http://localhost:8501/)
     - Network URL: [http://192.168.1.16:8501](http://192.168.1.16:8501/)

6. Upload an image and wait for the results

![Running Demo](https://github.com/duongstudent/Skin-Segmentation-and-Classification/raw/main/Results/SomeExample.png)

Benign Skin Demo Dataset Results:

![Benign Skin Demo Results](https://github.com/duongstudent/Skin-Segmentation-and-Classification/raw/main/Data_demo/benign_skin_demo_results.png)

Malignant Skin Demo Results:

![Malignant Skin Demo Results](https://github.com/duongstudent/Skin-Segmentation-and-Classification/raw/main/Data_demo/malignant_skin_demo_results.png)

## 2. Running the Segmentation Model Training Code

To run the code for training the segmentation model:

1. Navigate to the 'Training_Segformers' folder

2. Run the entire 'ISIC2020_Training_Segformers.ipynb' notebook
   - Make sure to download the ISIC 2018 dataset (link: https://challenge.isic-archive.com/data/#2018)
   - The code is pre-configured to run on Google Colab
   - Evaluate the results at the end of the notebook

## 3. Running the Classification Model Training Code

To run the code for training the classification model:

1. Navigate to the 'Training_ConvNeXt' folder

2. Download the ISIC 2020 dataset (link: https://challenge.isic-archive.com/data/#2020)

3. Split the dataset by running 'ISIC2020_data_split.ipynb'

4. Crop images by running 'ISIC_2020_segment_cropt.ipynb'

5. Augment data by running 'ISIC_2020_augment.ipynb'

6. Train and evaluate the model by running 'Training_ConvNeXt/ISIC2020_Training_ConvNeXt.ipynb'