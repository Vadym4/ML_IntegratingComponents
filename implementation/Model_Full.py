import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import DistilBertTokenizer
import torch
import torch.nn as nn
from torchvision import models as vision_models, transforms
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from torchvision import models

# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================ Screenshot Model (EfficientNet) ============================ #

class EfficientNetB0Model(nn.Module):
    def __init__(self, output_dim=256):
        super(EfficientNetB0Model, self).__init__()
        self.efficientnet = vision_models.efficientnet_b0(pretrained=True)
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(self.efficientnet.classifier[1].in_features, output_dim)
        )
     
      

    def forward(self, x):
        return self.efficientnet(x)

# ============================= URL Model ============================= #
class TextCNN(tf.keras.Model):
    def __init__(self, char_ngram_vocab_size, word_ngram_vocab_size, char_vocab_size,
                 word_seq_len, char_seq_len, embedding_size, l2_reg_lambda=0,
                 filter_sizes=[3, 4, 5, 6], dropout_rate=0.5):
        super(TextCNN, self).__init__()

        # Embedding layer for character input
        self.char_embedding = tf.keras.layers.Embedding(input_dim=char_vocab_size, output_dim=embedding_size)

        # Convolutional and pooling layers for character embeddings
        self.conv_layers = [
            tf.keras.layers.Conv2D(
                filters=256,
                kernel_size=(filter_size, embedding_size),
                activation='relu',
                padding="valid"
            ) for filter_size in filter_sizes
        ]

        self.pooling_layers = [
            tf.keras.layers.MaxPooling2D(
                pool_size=(char_seq_len - filter_size + 1, 1),
                padding="valid"
            ) for filter_size in filter_sizes
        ]

        # Fully connected layers
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')

        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, return_embeddings=False):
        # Character embeddings
        char_x = self.char_embedding(inputs)

        # Add channel dimension
        char_x_expanded = tf.expand_dims(char_x, -1)

        # Apply convolution and pooling
        pooled_outputs = [
            self.pooling_layers[i](self.conv_layers[i](char_x_expanded)) for i in range(len(self.conv_layers))
        ]
        pooled_outputs = tf.concat(pooled_outputs, axis=-1)

        # Flatten the output with a fixed shape
        flat_output = tf.reshape(
            pooled_outputs,
            [-1, 256 * len(self.conv_layers)]  # Ensure the flattened size matches convolution filters
        )

        # Dropout and fully connected layers
        dense_output1 = self.dense1(flat_output)
        dense_output2 = self.dense2(dense_output1)
        dense_output3 = self.dense3(dense_output2)

        if return_embeddings:
            return dense_output3, flat_output  # Return embeddings for concatenation
        return dense_output3

# ============================ Email Model (CNN) ============================ #


class EmailCNNModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_dim_output=64, kernel_size=3, num_filters=100):
        super(EmailCNNModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.conv1d = tf.keras.layers.Conv1D(num_filters, kernel_size, activation='relu', padding='same')
        self.global_max_pool = tf.keras.layers.GlobalMaxPooling1D()
        self.embedding_layer = tf.keras.layers.Dense(embedding_dim_output, activation='relu')  # Embeddings
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')  # Logits

    def call(self, x):
        x = self.embedding(x)
        x = self.conv1d(x)
        x = self.global_max_pool(x)
        embeddings = self.embedding_layer(x)  # Extract embeddings
        return embeddings  # Return embeddings for concatenation

    def classify(self, x):
        """Separate method for classification"""
        embeddings = self.call(x)
        return self.output_layer(embeddings)  # Logits for individual model use





class TextCNNClassifier(tf.keras.Model):
    def __init__(self, base_model):
        super(TextCNNClassifier, self).__init__()
        self.base_model = base_model

    def call(self, x):
        embeddings = self.base_model.call(x)  # Get embeddings from the base model
        return self.base_model.output_layer(embeddings)  # Pass embeddings through the classification layer


class EmailCNNClassifier(tf.keras.Model):
    def __init__(self, base_model):
        super(EmailCNNClassifier, self).__init__()
        self.base_model = base_model

    def call(self, x):
        embeddings = self.base_model.call(x)  # Get embeddings from the base model
        return self.base_model.output_layer(embeddings)  # Pass embeddings through the classification layer
    
 #------------------just for the single evaluation--------------------------#   
class ScreenshotClassifier(nn.Module):
    def __init__(self, base_model):
        super(ScreenshotClassifier, self).__init__()
        self.base_model = base_model  # Original ScreenshotModel
        self.classifier = nn.Linear(256, 1)  # Add classification layer

    def forward(self, x):
        embeddings = self.base_model(x)  # Use the original model to get embeddings
        logits = self.classifier(embeddings)  # Map embeddings to logits
        return logits.view(-1)  # Ensure outputs are 1D (batch size,)


def print_most_common_tokens_distilbert(tokenizer, tokenized_texts, num_tokens=10):
    """
    Prints the most common tokens in words after tokenizing with DistilBERT tokenizer.
    Args:
        tokenizer: The DistilBERT tokenizer.
        tokenized_texts: List of tokenized texts (output of tokenizer).
        num_tokens: Number of most common tokens to print.
    """
    from collections import Counter

    # Flatten all tokenized sequences into a single list of tokens
    all_tokens = [token for tokens in tokenized_texts['input_ids'] for token in tokens]

    # Count token frequencies
    token_counts = Counter(all_tokens)

    # Get the most common tokens and decode them
    print(f"\nTop {num_tokens} most common tokens:")
    for i, (token_id, count) in enumerate(token_counts.most_common(num_tokens), 1):
        decoded_token = tokenizer.decode([token_id]).strip()  # Decode token ID to string
        print(f"{i}. {decoded_token}: {count} occurrences")

#Preprocessing
def clean_email_text(email):
    import re
    email = re.sub(r'<[^>]+>', ' ', email)
    email = re.sub(r'http\S+|www\S+', 'URL', email)
    email = re.sub(r'\S+@\S+', 'EMAIL', email)    
    email = re.sub(r'\s+', ' ', email).strip()
    return email.lower()

def generate_url_model(vocab_size, max_len):
    char_ngram_vocab_size = 5000
    word_ngram_vocab_size = 10000
    char_vocab_size = vocab_size  
    word_seq_len = max_len
    char_seq_len = max_len
    embedding_size = 32
    filter_sizes = [3, 4, 5, 6]
    dropout_rate = 0.5
    l2_reg_lambda = 0.0

    # Create and return the TextCNN model
    model = TextCNN(
        char_ngram_vocab_size=char_ngram_vocab_size,
        word_ngram_vocab_size=word_ngram_vocab_size,
        char_vocab_size=char_vocab_size,
        word_seq_len=word_seq_len,
        char_seq_len=char_seq_len,
        embedding_size=embedding_size,
        l2_reg_lambda=l2_reg_lambda,
        filter_sizes=filter_sizes,
        dropout_rate=dropout_rate
    )

    return model


# ============================ Combined Model  ============================ #
def build_combined_model(url_vocab_size, email_vocab_size, url_shape, mail_shape, screenshot_shape):
    # Inputs            
    url_input = tf.keras.Input(shape=(url_shape,))
    email_input = tf.keras.Input(shape=(mail_shape,))
    screenshot_embedding_input = tf.keras.Input(shape=(screenshot_shape,))

    # URL TextCNN
    url_model = generate_url_model(vocab_size=url_vocab_size, max_len=url_shape)
    url_output = url_model(url_input, return_embeddings=True)[1]  # Get embeddings

    # Ensure fixed shape for URL model output
    url_output = layers.Dense(256, activation='relu')(url_output)  # Reduce to 256 dimensions for compatibility

    # Email CNN Model
    email_model = EmailCNNModel(vocab_size=email_vocab_size, embedding_dim=32)
    email_output = email_model(email_input)

    # Ensure fixed shape for Email model output
    email_output = layers.Dense(256, activation='relu')(email_output)  # Match dimensions to 256

    # Screenshot input is assumed to already have fixed embedding size (e.g., 256)
    screenshot_embedding = layers.Dense(256, activation='relu')(screenshot_embedding_input)  # Match dimensions to 256

    # Attention mechanism
    raw_attention_weights = layers.Dense(3, activation='softmax')(
        layers.concatenate([url_output, email_output, screenshot_embedding])
    )
    min_weight = 0.1
    adjusted_attention_weights = raw_attention_weights + min_weight
    normalized_attention_weights = tf.keras.layers.Lambda(lambda x: x / tf.reduce_sum(x, axis=1, keepdims=True))(
        adjusted_attention_weights
    )
    url_weight, mail_weight, screenshot_weight = tf.split(normalized_attention_weights, num_or_size_splits=3, axis=1)

    # Apply attention weights
    weighted_url = layers.multiply([url_output, url_weight])
    weighted_mail = layers.multiply([email_output, mail_weight])
    weighted_screenshot = layers.multiply([screenshot_embedding, screenshot_weight])

    # Concatenate embeddings
    concatenated_embeddings = layers.concatenate([weighted_url, weighted_mail, weighted_screenshot])

    # Final Classification Layers
    dense_layer = layers.Dense(64, activation='relu')(concatenated_embeddings)
    output_layer = layers.Dense(1, activation='sigmoid')(dense_layer)

    # Build Model
    model = tf.keras.Model(inputs=[url_input, email_input, screenshot_embedding_input], outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ============================ Data Preprocessing Functions ============================ #
def preprocess_urls(urls, tokenizer, max_len=200):
    sequences = tokenizer.texts_to_sequences(urls)
    return pad_sequences(sequences, maxlen=max_len)

def preprocess_emails(emails, tokenizer):
    encoded_emails = tokenizer(emails, padding=True, truncation=True, return_tensors='pt')
    print_most_common_tokens_distilbert(tokenizer, encoded_emails)
    return encoded_emails

import hashlib

def preprocess_screenshots(screenshots, screenshot_model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    processed_images = []
    total_images = len(screenshots)
    embedding_dim = 256  # Adjust according to the output dimension of ScreenshotModel
    cache_dir = "processed_tensors"  # Directory to store processed tensors
    os.makedirs(cache_dir, exist_ok=True)

    if screenshot_model is None:
        # Use default embeddings for testing
        print("Returning default embeddings for testing purposes...")
        default_embeddings = np.zeros((total_images, embedding_dim))
        return default_embeddings

    print("Loading and processing screenshots...")

    for idx, path in enumerate(screenshots):
        # Create a unique key based on the full path
        tensor_key = hashlib.md5(path.encode()).hexdigest()  # Unique hash for the path
        tensor_path = os.path.join(cache_dir, f"{tensor_key}.pt")

        if os.path.exists(tensor_path):
            # Load preprocessed tensor if available
            processed_image = torch.load(tensor_path)
        else:
            # Process the image if tensor not cached
            if os.path.exists(path):
                try:
                    image = Image.open(path).convert('RGB')
                    processed_image = transform(image)
                    # Save the tensor for future use
                    torch.save(processed_image, tensor_path)
                except UnidentifiedImageError:
                    processed_image = torch.zeros((3, 224, 224))
            else:
                print(f"Image not found at {path}, using default embedding.")
                processed_image = torch.zeros((3, 224, 224))

        processed_images.append(processed_image)

        # Progress update
        if (idx + 1) % (total_images // 10) == 0 or (idx + 1) == total_images:
            print(f"Processed {((idx + 1) / total_images) * 100:.0f}% of images")

    # Stack and move to device
    image_tensor = torch.stack(processed_images).to(device)

    # Run through ScreenshotModel to get embeddings
    batch_size = 32  # Adjust the batch size as needed for memory efficiency
    screenshot_embeddings = []
    for i in tqdm(range(0, total_images, batch_size), desc="Generating embeddings", unit="batch"):
        batch_tensor = image_tensor[i:i + batch_size]
        with torch.no_grad():
            batch_embeddings = screenshot_model(batch_tensor).cpu().numpy()
            screenshot_embeddings.append(batch_embeddings)

    # Concatenate all batches into a single array
    screenshot_embeddings = np.concatenate(screenshot_embeddings, axis=0)
    return screenshot_embeddings


# ============================ Training and Evaluation Functions ============================ #
def train_model(model, urls, emails, screenshots, labels, epochs=15, batch_size=32):
    labels = np.array(labels)  # Convert labels to NumPy array
    model.fit([urls, emails, screenshots], labels, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    # Save the model after training
    model.save("one_phase_model")
    print("Model saved as 'one_phase_model' in the current directory.")

from sklearn.metrics import roc_auc_score
def evaluate_model(model, urls, emails, screenshots, true_labels):
    predictions = model.predict([urls, emails, screenshots])
    predictions_binary = (predictions > 0.5).astype(int)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predictions_binary)
    precision = precision_score(true_labels, predictions_binary)
    recall = recall_score(true_labels, predictions_binary)
    f1 = f1_score(true_labels, predictions_binary)
    
    # Calculate AUC
    auc = roc_auc_score(true_labels, predictions)
    
    # Confusion Matrix
    #cm = confusion_matrix(true_labels, predictions_binary)
    #ConfusionMatrixDisplay(cm).plot(cmap=plt.cm.Blues)
    #plt.show()
    
    # Print metrics
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc}")
    log_model_performance("combined", accuracy, precision, recall, f1, auc, "results.txt")


def train_individual_model(model, data, labels, model_type=""):
    """Train an individual model on the respective data."""
    print(f"Training {model_type} model...")
    
    # Convert data and labels to NumPy arrays
    data = np.array(data) if not isinstance(data, np.ndarray) else data
    labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels

    # Split data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, test_size=0.1, random_state=42
    )

    # Fit the model
    model.fit(train_data, train_labels, 
              validation_data=(val_data, val_labels),
              epochs=10, batch_size=32)
    print(f"{model_type} model training complete.")
    return model

def evaluate_individual_model(model, data, labels, model_type=""):
    """Evaluate an individual model on test data."""
    print(f"Evaluating {model_type} model...")

    # Ensure `data` is NumPy array for TensorFlow compatibility
    data = np.array(data)

    # Get predictions
    predictions = model(data).numpy().squeeze()  # Call the model directly
    predictions_binary = (predictions > 0.5).astype(int)  # Binary thresholding

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions_binary)
    precision = precision_score(labels, predictions_binary)
    recall = recall_score(labels, predictions_binary)
    f1 = f1_score(labels, predictions_binary)
    auc = roc_auc_score(labels, predictions)

    print(f"{model_type} Model Performance:")
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc}")
    log_model_performance(model_type, accuracy, precision, recall, f1, auc, "Lemmatization_main.txt")

    return accuracy, precision, recall, f1, auc


# Screenshot model training
def train_screenshot_model(classifier_model, train_screenshots, train_labels, epochs=10, batch_size=32):
    classifier_model.train()
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()  # For raw logits

    # Preprocess screenshots
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset preparation
    processed_images = []
    for path in train_screenshots:
        if os.path.exists(path):
            try:
                image = Image.open(path).convert('RGB')
                processed_images.append(transform(image))
            except UnidentifiedImageError:
                processed_images.append(torch.zeros((3, 224, 224)))
        else:
            processed_images.append(torch.zeros((3, 224, 224)))

    # Stack and move to device
    screenshot_data = torch.stack(processed_images).to(device)
    screenshot_labels = torch.tensor(train_labels).float().to(device)  # Float for BCEWithLogitsLoss

    # Training loop
    for epoch in range(epochs):
        for i in range(0, len(screenshot_data), batch_size):
            batch_images = screenshot_data[i:i + batch_size]
            batch_labels = screenshot_labels[i:i + batch_size]

            optimizer.zero_grad()
            outputs = classifier_model(batch_images)  # Output logits
            loss = criterion(outputs, batch_labels)  # Compute loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


# Screenshot-only evaluation
# Screenshot evaluation
def evaluate_screenshot_model(screenshot_model, test_screenshots, test_labels, batch_size=32):
    print("Evaluating screenshot model in isolation...")
    
    # Preprocess screenshots into image tensors
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    processed_images = []
    for path in test_screenshots:
        if os.path.exists(path):
            try:
                image = Image.open(path).convert('RGB')
                processed_images.append(transform(image))
            except UnidentifiedImageError:
                # Use a blank image tensor for invalid images
                processed_images.append(torch.zeros((3, 224, 224)))
        else:
            # Use a blank image tensor for missing files
            processed_images.append(torch.zeros((3, 224, 224)))

    # Stack all processed images into a single tensor
    screenshot_data_tensor = torch.stack(processed_images).to(device)
    
    # Set model to evaluation mode
    screenshot_model.eval()

    # Prepare for batch processing
    all_outputs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(screenshot_data_tensor), batch_size), desc="Evaluating batches"):
            batch_data = screenshot_data_tensor[i:i + batch_size]
            outputs = screenshot_model(batch_data)  # Directly use logits
            all_outputs.append(outputs.cpu())

    # Concatenate all outputs
    outputs = torch.cat(all_outputs).numpy()

    # Binary classification threshold
    predictions_binary = (outputs > 0.5).astype(int)

    # Ensure `test_labels` is a flat array
    test_labels = np.array(test_labels).flatten()
    print("test_labels shape:", test_labels.shape)
    print("test_labels example:", test_labels[:5])
    print("predictions_binary shape:", predictions_binary.shape)
    print("predictions_binary example:", predictions_binary[:5])

    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions_binary)
    precision = precision_score(test_labels, predictions_binary)
    recall = recall_score(test_labels, predictions_binary)
    f1 = f1_score(test_labels, predictions_binary)
    auc = roc_auc_score(test_labels, outputs)

    print("Screenshot Model Performance:")
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc}")
    log_model_performance("Screenshots", accuracy, precision, recall, f1, auc, "Lemmatization_main.txt")




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
# ============================ Main Script ============================ #


def main():
    results_file = "results.txt"
    if os.path.exists(results_file):
        os.remove(results_file)


    phishing_data = pd.read_csv("own_dataset/Mil_Naz.csv")
    clean_data = pd.read_csv("own_dataset/unique_merged_nohardham.csv")


    phishing_data['label'] = 1
    clean_data['label'] = 0
    all_data = pd.concat([phishing_data, clean_data], ignore_index=True)

    # Handle missing values
    all_data['URL'] = all_data['URL'].fillna('')
    all_data['email_text'] = all_data['email_text'].fillna('')
    all_data['screenshot'] = all_data['screenshot'].fillna('')

    # Map screenshots based on label
    print("Mapping screenshot paths...")
    
    all_data['screenshot'] = all_data.apply(
        lambda row: f"own_dataset/screenshots_phishing/{row['screenshot']}" if row['label'] == 1 
                    else f"own_dataset/filtered_unique_screenshots/{row['screenshot']}",
        axis=1
    )
    all_data = all_data.replace(to_replace=['<br />','br', 'jose@monkey.org', 'jose', 'MillerSmiles', 'millersmiles', 'co', 'uk', 'millersmiles.co.uk', 'Phishing scam report', '\ufeff'], value='', regex=True)
    all_data['email_text'] = all_data['email_text'].fillna('').apply(clean_email_text)
    
    for run in range(1, 12):
        print("Loading data...")
        
        # Split data into training and testing sets
        all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
        split_index = int(0.7 * len(all_data)) #PARAMETER: Train-Test-split
        train_data = all_data[:split_index]

        test_data = all_data[split_index:]  
            # Extract features and labels
        train_urls = train_data['URL'].tolist()
        train_emails = train_data['email_text'].tolist()
        train_screenshots = train_data['screenshot'].tolist()
        train_labels = train_data['label'].tolist()
        
        test_urls = test_data['URL'].tolist()
        test_emails = test_data['email_text'].tolist()
        test_screenshots = test_data['screenshot'].tolist()
        test_labels = test_data['label'].tolist()
        from collections import Counter
        print("Test Labels Distribution:", Counter(test_labels))
        
        # URL preprocessing
        print("Preprocessing URLs...")
        url_tokenizer = Tokenizer(char_level=True)
        url_tokenizer.fit_on_texts(train_urls)
        train_url_data = preprocess_urls(train_urls, url_tokenizer)

        # Email preprocessing with DistilBERT tokenizer
        print("Preprocessing emails with DistilBERT tokenizer...")
        email_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        train_email_data = preprocess_emails(train_emails, email_tokenizer)

        # Screenshot model initialization and preprocessing
        screenshot_model = EfficientNetB0Model(output_dim=256).to(device)
        train_screenshot_data = preprocess_screenshots(train_screenshots, screenshot_model)
        
        # Convert email and screenshot data to NumPy for TensorFlow compatibility
        train_email_data = train_email_data['input_ids'].cpu().numpy() if isinstance(train_email_data['input_ids'], torch.Tensor) else train_email_data['input_ids']
        train_screenshot_data = train_screenshot_data if isinstance(train_screenshot_data, np.ndarray) else train_screenshot_data.cpu().numpy()


        # Build and Train Combined Model
        print("Building model...")
        combined_model = build_combined_model(
            url_vocab_size=len(url_tokenizer.word_index) + 1,
            email_vocab_size=len(email_tokenizer.vocab),
            url_shape=train_url_data.shape[1],
            mail_shape=train_email_data.shape[1],
            screenshot_shape=256  # Fixed integer shape for screenshot embedding input
        )

        print("Training model...")
        train_model(combined_model, train_url_data, train_email_data, train_screenshot_data, train_labels)
        print("Training complete.")
        

        test_url_data = preprocess_urls(test_urls, url_tokenizer)

        test_email_data = preprocess_emails(test_emails, email_tokenizer)
        test_screenshot_data = preprocess_screenshots(test_screenshots, screenshot_model)
        test_email_data = test_email_data['input_ids'].cpu().numpy() if isinstance(test_email_data['input_ids'], torch.Tensor) else test_email_data['input_ids']
        test_screenshot_data = test_screenshot_data if isinstance(test_screenshot_data, np.ndarray) else test_screenshot_data.cpu().numpy()

        print("Evaluating model...")
        evaluate_model(combined_model, test_url_data, test_email_data, test_screenshot_data, test_labels)
        print("Evaluation complete.")

     


if __name__ == "__main__":
    main()


