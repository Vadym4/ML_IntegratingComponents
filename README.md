# ML_IntegratingComponents
This repository contains the Dataset and Code to the work "A Novel Approach to Improve Machine Learning-Based Phishing Email Detection:  Integrating Text, URL, and Website  Components"

# Dataset (Folder: Dataset)
This folder contains the dataset constructed as part of this work
- Each datset is structured as csv file, with columns email_text, url and screenshot. The screenshot column contains the filename of the corresponding screenshots
- Dataset 1: Phishing (based on MillerSmiles and Nazario). 
- Dataset 2: Clean (based on SpamAssassin, CEAS, TREC-7 and Ling). 
- Dataset 3: HardHam1 (HardHam-Subset 1 described in the work, based on SpamAssassin).  
- Dataset 4: HardHam2 (HardHam-Subset 2 described in the work, based manually filtered samples from CEAS).
- Dataset 5: HardHam3 (HardHam-Subset 3 described in the work, based on crawled Screenshots from WayBack Machine + Artificially generated email texts). 

# Implementation (Folder: Implementation)
This folder contains the implementation of the model, which was created in this work
- Model_Full.py: One File, which contains the model implementation, preprocessing, training and evaluation. Is currently being refactored in multiple files for better usability
  
