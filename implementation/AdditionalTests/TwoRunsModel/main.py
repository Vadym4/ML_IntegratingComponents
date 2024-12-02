import numpy as np
from helper_functions import * 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_embeddings():
    # Load embeddings from each component, MUST be the same file in the helper_function! 
    url_embeddings = np.load('urls/embeddings_final.npy')
    mail_embeddings = np.load('email_extended/final_email_embeddings.npy')
    screenshot_embeddings = np.load('screenshots/screenshot_embeddings.npy')


    # Determine the maximum number of samples across all embeddings
    max_samples = max(url_embeddings.shape[0], mail_embeddings.shape[0], screenshot_embeddings.shape[0])

    # Pad the smaller dataset(s) to match the largest one in terms of sample size (number of rows)
    if mail_embeddings.shape[0] < max_samples:
        padding = np.zeros((max_samples - mail_embeddings.shape[0], mail_embeddings.shape[1]))
        mail_embeddings = np.vstack([mail_embeddings, padding])

    if url_embeddings.shape[0] < max_samples:
        padding = np.zeros((max_samples - url_embeddings.shape[0], url_embeddings.shape[1]))
        url_embeddings = np.vstack([url_embeddings, padding])

    if screenshot_embeddings.shape[0] < max_samples:
        padding = np.zeros((max_samples - screenshot_embeddings.shape[0], screenshot_embeddings.shape[1]))
        screenshot_embeddings = np.vstack([screenshot_embeddings, padding])

    return url_embeddings, mail_embeddings, screenshot_embeddings



import tensorflow as tf
from tensorflow.keras import layers

def build_combined_model(input_shape):
    # Input shape is the concatenated embedding shape
    input_layer = layers.Input(shape=input_shape)

    # Add a dense layer (linear layer) with ReLU activation
    dense_layer = layers.Dense(128, activation='relu')(input_layer)

    # Add a second dense layer
    dense_layer_2 = layers.Dense(64, activation='relu')(dense_layer)


    # Output layer for binary classification (or softmax for multi-class)
    output_layer = layers.Dense(1, activation='sigmoid')(dense_layer_2)

    # Build the model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with an optimizer and loss function
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


from tensorflow.keras import layers, models

import tensorflow as tf
from tensorflow.keras import layers, models

def build_combined_model_with_attention(url_shape, mail_shape, screenshot_shape):
    # Separate input layers for each embedding
    url_input = layers.Input(shape=url_shape, name='url_input')
    mail_input = layers.Input(shape=mail_shape, name='mail_input')
    screenshot_input = layers.Input(shape=screenshot_shape, name='screenshot_input')
    
    # Dense layers to process each embedding individually
    url_dense = layers.Dense(128, activation='relu')(url_input)
    mail_dense = layers.Dense(128, activation='relu')(mail_input)
    screenshot_dense = layers.Dense(128, activation='relu')(screenshot_input)
    
    # Attention weights with minimum constraint
    raw_attention_weights = layers.Dense(3, activation='softmax', name='raw_attention_weights')(layers.concatenate([url_dense, mail_dense, screenshot_dense]))
    
    # Add 0.1 minimum weight and renormalize to sum to 1
    min_weight = 0.1
    adjusted_attention_weights = raw_attention_weights + min_weight
    normalized_attention_weights = tf.keras.layers.Lambda(lambda x: x / tf.reduce_sum(x, axis=1, keepdims=True), name='attention_weights')(adjusted_attention_weights)
    
    # Split the adjusted weights
    url_weight, mail_weight, screenshot_weight = tf.split(normalized_attention_weights, num_or_size_splits=3, axis=1)
    
    # Apply weights to embeddings
    weighted_url = layers.multiply([url_dense, url_weight])
    weighted_mail = layers.multiply([mail_dense, mail_weight])
    weighted_screenshot = layers.multiply([screenshot_dense, screenshot_weight])
    
    # Concatenate the weighted embeddings
    concatenated_embeddings = layers.concatenate([weighted_url, weighted_mail, weighted_screenshot])
    
    # Final dense layers for classification
    dense_layer = layers.Dense(64, activation='relu')(concatenated_embeddings)
    output_layer = layers.Dense(1, activation='sigmoid', name='classification_output')(dense_layer)
    
    # Build and compile the model with losses specified for each output
    model = models.Model(inputs=[url_input, mail_input, screenshot_input], outputs=[output_layer, normalized_attention_weights])
    model.compile(
        optimizer='adam', 
        loss={'classification_output': 'binary_crossentropy', 'attention_weights': None},  # Set loss for attention_weights to None
        metrics={'classification_output': 'accuracy'}  # Only track metrics for classification output
    )
    
    return model



def train_combined_model(combined_model):
    # Load the new dataset (containing URLs, emails, and labels)
    urls, emails, pictures, labels = load_dataset("train")

    # Generate embeddings on the fly using pre-trained models
    url_embeddings = generate_url_embeddings(urls)
    mail_embeddings = generate_email_embeddings(emails)
    screenshot_embeddings = generate_screenshot_embeddings(pictures)

    labels = np.array(labels)

    # Train the model, specifying only the target for the classification output
    combined_model.fit(
        [url_embeddings, mail_embeddings, screenshot_embeddings], 
        {"classification_output": labels},  # Pass target only for classification
        epochs=20, 
        batch_size=32, 
        validation_split=0.1
    )

    # Save the model after training
    combined_model.save('combined_model_with_attention.h5')


from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt


def save_results(file_path, content):
    with open(file_path, 'a') as file:
        file.write(content + '\n')

# Function to evaluate and log performance
def log_model_performance(model_type, accuracy, precision, recall, f1, auc, file_path):
    results = f"""
Run - {model_type} Model Performance:
-------------------------------------------
Accuracy: {accuracy}
Precision: {precision}
Recall: {recall}
F1 Score: {f1}
AUC: {auc}
"""    
    save_results(file_path, results)


def test_combined_model():
    # Load the test dataset (containing URLs, emails, and labels)
    urls, emails, screenshots, test_labels = load_dataset("test")

    # Generate embeddings for the test data
    url_embeddings = generate_url_embeddings(urls)
    mail_embeddings = generate_email_embeddings(emails)
    screenshot_embeddings = generate_screenshot_embeddings(screenshots)

    # Load the trained combined model
    combined_model = tf.keras.models.load_model('combined_model_with_attention.h5')

    # Make predictions on the test set, capturing all outputs
    all_outputs = combined_model.predict([url_embeddings, mail_embeddings, screenshot_embeddings])

    # Access the classification predictions and attention weights
    predictions = all_outputs[0]  # Classification output
    attention_weights = all_outputs[1]  # Attention weights output

    # Convert probabilities to binary labels (0 or 1)
    predictions = (predictions > 0.5).astype(int)

    # Print the attention weights for each test sample
    print("Attention Weights for Test Samples:")
    for i, weights in enumerate(attention_weights):
        print(f"Sample {i+1}: URL Weight = {weights[0]}, Mail Weight = {weights[1]}, Screenshot Weight = {weights[2]}")

    # Step 6: Evaluate metrics (accuracy, precision, recall, F1-score, AUC)
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    auc = roc_auc_score(test_labels, predictions)

    # Step 7: Print the evaluation metrics
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test AUC Score: {auc:.4f}")
    log_model_performance("Combined", accuracy, precision, recall, f1, auc, 'TwoRuns_results.txt')

    # Step 8: Generate and display confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    pd.set_option('display.max_colwidth', None)

    # Identify misclassified samples and print the first 100 characters of each misclassified email
    #print(predictions)
    #print(test_labels)
    predictions = predictions.flatten()

    misclassified_indices = np.where(predictions != test_labels)[0]
    #print("\nFirst 100 characters of each misclassified email:")

    #for i in misclassified_indices:
    #    print(f"Sample {i+1}: {emails[i][:100]}")  # Print first 100 characters
    #    print(screenshots[i])




'''
#Phase1: Train each model seperately
#For now straight calls with default datasets, later parameters to specify datasets 

def train_phase1(): #call from command line
    #calculate_embeddings_website()
    #calculate_embeddings_mail()
    calculate_embeddings_url()

import subprocess
def calculate_embeddings_url(): 
    #Part 1: Train the url-model. This will be the URL-NetModel with a few adaptions so it runs with python 2 Everything there is already prepared, only at a later timepoint the possible to do parameters (e.g. for the training data) will be added
    command = [
        'python', 'url/train.py', 
#        '--data.data_dir', 'baseline_models/urlnet-master/data/dataset_urlnet/urldata_formatted.txt',
    ]
    subprocess.run(command)
    
    #Part 2: Train the mail-model. This is a simple 1D model. 
    command = [
        'python', 'emails/embeddings_emails.py', 
#        '--data.data_dir', 'baseline_models/urlnet-master/data/dataset_urlnet/urldata_formatted.txt',
    ]
    subprocess.run(command)

    #Part 3: Train the screenshot model 
'''


from tensorflow.keras.utils import plot_model


def train_phase2(): 
    # Load and concatenate pre-trained embeddings
    url_embeddings, mail_embeddings, screenshot_embeddings = load_embeddings()
    
    # Build the combined model using the shape of the individual embeddings
    combined_model = build_combined_model_with_attention(url_embeddings.shape[1], 
                                                         mail_embeddings.shape[1], 
                                                         screenshot_embeddings.shape[1])

    # Train the combined model with attention on the new data (URLs, emails, and labels)
    train_combined_model(combined_model)



def test():
    test_combined_model()
    

#------------------M  A  I  N     P  I  P  E  L  I  N  E  ------------------------
#pipeline: Run train_phase1, then trainphase2, then test 

#train_phase1() #=> only necassary once, then its saved as pretrained model. As the data is unrelated to the train/test data, no need to rerun this. 
train_phase2() #=> trains the second phase
test()