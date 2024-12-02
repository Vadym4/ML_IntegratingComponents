
import tensorflow as tf
tf.compat.v1.enable_eager_execution()  # Move this line to the top
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd

import torch.nn as nn


###################################### TextCNN Model ###########################################
class TextCNN(tf.keras.Model):
    def __init__(self, char_ngram_vocab_size, word_ngram_vocab_size, char_vocab_size,
                 word_seq_len, char_seq_len, embedding_size, l2_reg_lambda=0,
                 filter_sizes=[3, 4, 5, 6], dropout_rate=0.5):
        super(TextCNN, self).__init__()

        # Hyperparameters
        self.l2_loss = tf.constant(0.0)
        self.dropout_rate = dropout_rate

        # Embedding Layer for characters only (emb_mode=1)
        self.char_embedding = tf.keras.layers.Embedding(input_dim=char_vocab_size, output_dim=embedding_size)

        # Convolution and pooling layers for char embeddings
        self.conv_layers = [
            tf.keras.layers.Conv2D(
                filters=256,
                kernel_size=(filter_size, embedding_size),
                strides=(1, 1),
                padding="valid",
                activation='relu'
            ) for filter_size in filter_sizes
        ]

        # Pooling layers
        self.pooling_layers = [
            tf.keras.layers.MaxPooling2D(
                pool_size=(char_seq_len - filter_size + 1, 1),
                strides=(1, 1),
                padding='valid'
            ) for filter_size in filter_sizes
        ]

        # Fully connected Dense layers after pooling
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')

        # Final classification layer
        self.output_layer = tf.keras.layers.Dense(2)

        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)


    def call(self, inputs, return_embeddings=False):
        # Input is character sequence
        input_x_char_seq = inputs

        # Embedding lookup for char sequence
        embedded_x_char_seq = self.char_embedding(input_x_char_seq)

        # Add a channel dimension to the char embeddings for Conv2D
        embedded_x_char_seq_expanded = tf.expand_dims(embedded_x_char_seq, -1)

        # Apply convolution and pooling to char embeddings
        pooled_char_x = [self.pooling_layers[i](self.conv_layers[i](embedded_x_char_seq_expanded)) for i in range(len(self.conv_layers))]
        h_pool_char = tf.concat(pooled_char_x, 3)  # Combine the outputs from all pooling layers

        # Flatten the pooled output to 2D (batch_size, flattened_embedding)
        x_flat_char = tf.reshape(h_pool_char, [h_pool_char.shape[0], -1])

        # Apply Dropout
        h_drop_char = self.dropout(x_flat_char)

        # Fully connected layers and output
        output0 = self.dense1(h_drop_char)
        output1 = self.dense2(output0)
        output2 = self.dense3(output1)
        scores = self.output_layer(output2)

        if return_embeddings:
            return scores, x_flat_char  # Return the flattened embeddings for concatenation

        return scores


    def accuracy(self, labels, logits):
        # Accuracy calculation
        predictions = tf.argmax(logits, axis=1)
        correct_preds = tf.equal(predictions, tf.argmax(labels, axis=1))
        return tf.reduce_mean(tf.cast(correct_preds, "float"))

def tt_split():
    clean = "own_dataset/Mil_Naz.csv"
    phishing = "own_dataset/unique_merged_nohardham.csv"
    for input_file in [clean, phishing]:
        test_size = 0.3

        # Load the CSV file
        df = pd.read_csv(input_file)

        # Split the data into train and test sets
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

        print(len(train_df))
        # Generate output file names
        train_file = input_file.replace('.csv', '_train.csv')
        test_file = input_file.replace('.csv', '_test.csv')

        # Save the train and test data to new CSV files
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

def load_dataset(param):
    tt_split()
    # Load phishing samples
    if(param=="train"):

        #Commented out: Hard Ham Spammassassin Dataset
        #phishing_df = pd.read_csv('own_dataset/millersmiles_train.csv').head(35)
        #        clean_df = pd.read_csv('own_dataset/hardHam_spamassassin_train.csv')

        phishing_df = pd.read_csv('own_dataset/Mil_Naz_train.csv')
        clean_df = pd.read_csv('own_dataset/unique_merged_nohardham_train.csv') 
        #clean_df = pd.read_csv('own_dataset/hardHam_SpamAssassin.csv')

    else:
        #phishing_df = pd.read_csv('own_dataset/millersmiles_test.csv')
        phishing_df = pd.read_csv('own_dataset/Mil_Naz_test.csv')
        #clean_df = pd.read_csv('own_dataset/CAES_hardHAM.csv')
        clean_df = pd.read_csv('own_dataset/unique_merged_nohardham_test.csv')

        #clean_df = pd.read_csv('own_dataset/CAES_hardHAM.csv')
         #clean_df = pd.read_csv('own_dataset/hardHam_spamassassin_test.csv')
    
    dfs = [phishing_df, clean_df]
    for i in range(len(dfs)):
        dfs[i] = dfs[i].replace(to_replace=['<br />','br', 'jose@monkey.org', 'jose', 'MillerSmiles', 'millersmiles', 'co', 'uk', 'millersmiles.co.uk', 'Phishing scam report', '\ufeff'], value='', regex=True)
        dfs[i]['email_text']= dfs[i]['email_text'].replace(to_replace=['http'], value='', regex=True)
        dfs[i]['email_text'] = dfs[i]['email_text'].fillna('').apply(clean_email_text)
       
        
    phishing_df, clean_df = dfs
    # Extract URLs and emails from both datasets
    phishing_urls = phishing_df['URL'].tolist()
    phishing_emails = phishing_df['email_text'].tolist()
    phishing_pictures = phishing_df['screenshot'].tolist()
    phishing_pictures = ['screenshots_phishing/' + pic for pic in phishing_pictures]


    clean_urls = clean_df['URL'].tolist()
    print(len(clean_urls))
    print("______")

    clean_emails = clean_df['email_text'].tolist()
    clean_pictures = clean_df['screenshot'].tolist()

    if (param == "train"):
        clean_pictures = ['filtered_unique_screenshots/' + pic for pic in clean_pictures]
    else:
        #clean_pictures = ['caes_hardh_screenshots/' + pic for pic in clean_pictures]
        clean_pictures = ['filtered_unique_screenshots/' + pic for pic in clean_pictures]

    
    
    
    # Create labels: 1 for phishing, 0 for clean
    phishing_labels = [1] * len(phishing_df)
    clean_labels = [0] * len(clean_df)
    
    # Combine both datasets
    urls = phishing_urls + clean_urls
    emails = phishing_emails + clean_emails
    pictures = phishing_pictures + clean_pictures
    labels = phishing_labels + clean_labels
    
    return urls, emails, pictures,  labels

def generate_url_embeddings(urls):
    # Step 1: Rebuild the model architecture (same as the original model)
    char_ngram_vocab_size = 5000
    word_ngram_vocab_size = 10000
    char_vocab_size = 10000
    word_seq_len = 200
    char_seq_len = 200
    embedding_size = 32
    filter_sizes = [3, 4, 5, 6]
    dropout_rate = 0.5
    l2_reg_lambda = 0.0

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

    # Step 2: Call the model on dummy data to initialize variables
    dummy_input = np.zeros((1, char_seq_len))  # Batch size 1, sequence length = char_seq_len
    model(dummy_input)  # This creates the variables for the model

    model_path = 'urls/final_model.weights.h5'
    # Step 3: Load the saved weights into the model
    model.load_weights(model_path)

    # Step 4: Preprocess URLs to match the format expected by the model
    max_len_chars = 200  # Ensure it matches the model's max sequence length
    tokenizer = Tokenizer(char_level=True)  # Assuming the model uses character-level tokenization
    urls = [str(url) if url is not None else '' for url in urls] #Some mails dont have URL!
    tokenizer.fit_on_texts(urls)

    # Tokenize and pad the input URLs
    char_x = tokenizer.texts_to_sequences(urls)
    char_x_padded = pad_sequences(char_x, maxlen=max_len_chars)

    # Convert to numpy array
    char_x_padded = np.array(char_x_padded)

    # Step 5: Use the model to generate embeddings
    _, embeddings = model(char_x_padded, return_embeddings=True)

    # Return the generated embeddings
    return embeddings.numpy()




import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision import models, transforms

# Define the transformation for image preprocessing (ResNet input size is 224x224)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet18 expects input size of 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ResNet
])

class ScreenshotModel(nn.Module):
    def __init__(self, output_dim):
        super(ScreenshotModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, output_dim)
    
    def forward(self, x):
        x = self.resnet(x)
        return x
    

from PIL import Image, UnidentifiedImageError

def generate_screenshot_embeddings(pictures, model_path='screenshots/screenshot_model_mixed.pth', output_dim=256):
    """
    Generates embeddings for a list of pictures using the pre-trained ScreenshotModel.
    If a picture is not found or cannot be loaded, a default embedding is generated.
    

        pictures (list): List of picture file paths.
        model_path (str): Path to the pre-trained screenshot model.
        output_dim (int): Dimensionality of the output embeddings.
    Returns:
        embeddings (np.ndarray): 2D array of picture embeddings (num_pictures, embedding_dim).
    """
    #model_path = "voting_screenshots.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained screenshot model
    model = ScreenshotModel(output_dim=output_dim).to(device)
    model.eval()  # Set model to evaluation mode
    
    # Load the saved weights into the model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # List to store embeddings
    all_embeddings = []
    print("PICTURE COUNT")
    total_pictures = len(pictures)  # Get the total number of pictures

    # Iterate over the pictures and generate embeddings
    for i, picture_path in enumerate(pictures):
        picture_path = "own_dataset/" + picture_path
        try:
            # Load and preprocess the image
            image = Image.open(picture_path).convert('RGB')
            image_tensor = image_transform(image).unsqueeze(0)  # Add batch dimension
            
            image_tensor = image_tensor.to(device)
            
            # Generate embedding (hidden representation) from the model
            with torch.no_grad():
                embedding = model(image_tensor)

            # Store the embedding as a numpy array
            all_embeddings.append(embedding.cpu().numpy())
        
        except (FileNotFoundError, UnidentifiedImageError) as e:
            # Print error message
            print(f"Error loading {picture_path}: {e}")
            # Generate a default embedding (e.g., zeros) if the file is not found or cannot be opened
            default_embedding = np.zeros((1, output_dim))
            all_embeddings.append(default_embedding)

        # Progress print (percentage of pictures processed)
        processed_percent = (i + 1) / total_pictures * 100
        print(f"Processed {i + 1}/{total_pictures} pictures ({processed_percent:.2f}%)", end='\r')

    print("\nAll pictures processed.")
    
    # Convert list of embeddings to a 2D numpy array
    embeddings = np.vstack(all_embeddings)  # (num_pictures, embedding_dim)
    
    return embeddings








import torch
import torch.nn as nn
from transformers import DistilBertTokenizer


class CNNTextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes=1, kernel_size=3, num_filters=100):
        super(CNNTextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_filters, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # Change to [batch_size, embedding_dim, sequence_length] for Conv1d
        x = self.conv1d(x)
        x = torch.relu(x)
        x = self.global_max_pool(x).squeeze(2)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # For extracting embeddings (layer before final classification)
    def extract_embeddings(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = torch.relu(x)
        x = self.global_max_pool(x).squeeze(2)
        x = torch.relu(self.fc1(x))
        return x  # Return the extracted embeddings

def clean_email_text(email):
    import re
    email = re.sub(r'<[^>]+>', ' ', email)
    email = re.sub(r'http\S+|www\S+', 'URL', email)
    email = re.sub(r'\S+@\S+', 'EMAIL', email)
    email = re.sub(r'\s+', ' ', email).strip()
    return email.lower()

# Function to generate email embeddings from pre-trained model
def generate_email_embeddings(emails,  model_path='email_extended/simpleCNN.pth'): #emails/output_old/
    #model_path='voting_emails.pth'

    # Step 1: Load the tokenizer (DistilBertTokenizer in this case)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    max_len = 500
    tokenized_data = {'input_ids': [], 'attention_mask': []}

    # Step 2: Tokenize and pad the emails
    for text in emails:
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        tokenized_data['input_ids'].append(inputs['input_ids'].squeeze(0))
        tokenized_data['attention_mask'].append(inputs['attention_mask'].squeeze(0))

    tokenized_data['input_ids'] = torch.stack(tokenized_data['input_ids'])

    # Step 3: Load the pre-trained model
    vocab_size = tokenizer.vocab_size
    embedding_dim = 100
    model = CNNTextModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Step 4: Generate embeddings for emails
    input_ids = tokenized_data['input_ids'].to(device)
    with torch.no_grad():
        embeddings = model.extract_embeddings(input_ids)

    # Convert embeddings to numpy array and return
    return embeddings.cpu().numpy()


import torch
from transformers import DistilBertTokenizer

# Function to generate email predictions using the pre-trained CNN model => just for voting! 
def generate_email_predictions(emails, model_path='voting_emails.pth', max_len=500):

    model_path = 'emails/simpleCNN.pth'
    """
    Generate predictions for emails using the pre-trained CNNTextModel.
    
    Args:
        emails (list): List of email texts.
        model_path (str): Path to the trained PyTorch model.
        max_len (int): Maximum length for tokenizing email texts.
    
    Returns:
        predictions (list): List of binary predictions for each email.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the tokenizer (DistilBertTokenizer)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Tokenize and pad the emails
    tokenized_data = {'input_ids': [], 'attention_mask': []}
    for text in emails:
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        tokenized_data['input_ids'].append(inputs['input_ids'].squeeze(0))
        tokenized_data['attention_mask'].append(inputs['attention_mask'].squeeze(0))

    # Convert list to torch tensors
    tokenized_data['input_ids'] = torch.stack(tokenized_data['input_ids']).to(device)

    # Load the pre-trained CNN model
    vocab_size = tokenizer.vocab_size
    embedding_dim = 100
    model = CNNTextModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Generate predictions
    predictions = []
    with torch.no_grad():
        for input_ids in tokenized_data['input_ids']:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension
            outputs = model(input_ids).squeeze(-1)
            pred = torch.sigmoid(outputs).cpu().numpy()
            pred = (pred > 0.5).astype(int)  # Convert to binary prediction (0 or 1)
            predictions.append(pred)

    return predictions



import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def generate_url_predictions(urls, model_path='urls/best_model.weights.h5', max_len_chars=200): 
    """
    Generate predictions for URLs using the pre-trained TextCNN model.
    
    Args:
        urls (list): List of URLs.
        model_path (str): Path to the trained TensorFlow model weights.
        max_len_chars (int): Maximum sequence length for tokenized URLs.
    
    Returns:
        predictions (list): List of binary predictions for each URL.
    """
    # Step 1: Rebuild the model architecture (same as the original model)
    
    char_ngram_vocab_size = 5000
    word_ngram_vocab_size = 10000
    char_vocab_size = 10000
    word_seq_len = 200
    char_seq_len = 200
    embedding_size = 32
    filter_sizes = [3, 4, 5, 6]
    dropout_rate = 0.5
    l2_reg_lambda = 0.0

    # Initialize the TextCNN model
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

    # Step 2: Call the model on dummy data to initialize variables
    dummy_input = np.zeros((1, char_seq_len))  # Batch size 1, sequence length = char_seq_len
    model(dummy_input)  # This creates the variables for the model

    # Step 3: Load the saved weights into the model
    model.load_weights(model_path)

    # Step 4: Preprocess URLs to match the format expected by the model
    tokenizer = Tokenizer(char_level=True)  # Character-level tokenization
    tokenizer.fit_on_texts(urls)

    # Tokenize and pad the input URLs
    char_x = tokenizer.texts_to_sequences(urls)
    char_x_padded = pad_sequences(char_x, maxlen=max_len_chars)

    # Convert to numpy array
    char_x_padded = np.array(char_x_padded)

    # Step 5: Generate predictions from the model
    logits = model(char_x_padded)  # Get only the logits
    predictions = tf.argmax(logits, axis=1).numpy()  # Convert logits to binary predictions (0 or 1)

    return predictions





def training_url_model_phase2(urls, labels, model_path='urls/final_model.weights.h5'):
    # Step 1: Rebuild the model architecture (same as the original model)
    char_ngram_vocab_size = 5000
    word_ngram_vocab_size = 10000
    char_vocab_size = 10000
    word_seq_len = 200
    char_seq_len = 200
    embedding_size = 32
    filter_sizes = [3, 4, 5, 6]
    dropout_rate = 0.5
    l2_reg_lambda = 0.0

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

    # Step 2: Create a dummy input to initialize model variables
    dummy_input = np.zeros((1, char_seq_len))  # Batch size 1, sequence length = char_seq_len
    model(dummy_input)  # Initialize the model's variables by calling it once

    # Step 3: Load the pre-trained model weights
    model.load_weights(model_path)

    # Step 4: Preprocess URLs to match the format expected by the model
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(urls)
    char_x = tokenizer.texts_to_sequences(urls)
    char_x_padded = pad_sequences(char_x, maxlen=char_seq_len)

    # One-hot encode the labels to match the (batch_size, 2) output shape
    labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=2)  # Convert labels to one-hot encoding

    # Step 5: Continue training the model on the new data
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    batch_size = 32
    nb_epochs = 5

    dataset = tf.data.Dataset.from_tensor_slices((char_x_padded, labels_one_hot)).batch(batch_size)
    for epoch in range(nb_epochs):
        print(f"Training URL model - Epoch {epoch+1}/{nb_epochs}")
        for step, (x_batch, y_batch) in enumerate(dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch)
                loss_value = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_batch, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        print(f"Epoch {epoch+1} completed.")

    # Step 6: Save the updated model
    model_path = "voting_url.weights.h5"
    model.save_weights(model_path)
    print("URL model re-saved after additional training.")

import torch

def training_email_model_phase2(emails, labels, model_path='emails/simpleCNN.pth'):
    # Step 1: Tokenize and preprocess the emails 
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    max_len = 500
    
    tokenized_data = {'input_ids': [], 'attention_mask': []}
    for text in emails:
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        tokenized_data['input_ids'].append(inputs['input_ids'].squeeze(0))
        tokenized_data['attention_mask'].append(inputs['attention_mask'].squeeze(0))
    tokenized_data['input_ids'] = torch.stack(tokenized_data['input_ids'])

    # Step 2: Load the pre-trained email model
    vocab_size = tokenizer.vocab_size
    embedding_dim = 100
    model = CNNTextModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_path))
    model.train()

    # Step 3: Continue training with new data
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    batch_size = 32
    epochs = 5
    input_ids = tokenized_data['input_ids'].to(device)
    
    for epoch in range(epochs):
        model.train()
        for batch_idx in range(0, len(input_ids), batch_size):
            inputs = input_ids[batch_idx:batch_idx+batch_size]
            labels_batch = torch.tensor(labels[batch_idx:batch_idx+batch_size]).float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} completed.")

    model_path = 'voting_emails.pth'
    # Step 4: Save the updated model
    torch.save(model.state_dict(), model_path)
    print("Email model re-saved after additional training.")

import torch

import os
from PIL import Image

def training_screenshot_model_phase2(screenshots, labels, model_path='screenshots/screenshot_model_mixed.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained screenshot model
    model = ScreenshotModel(output_dim=256).to(device)
    classification_layer = torch.nn.Linear(256, 1).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    classification_layer.load_state_dict(checkpoint['classification_layer_state_dict'])
    
    # Preprocess the screenshots (e.g., using a transform pipeline)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    screenshot_tensors = []
    valid_labels = []

    # Load and process screenshots, skipping missing files
    for i, screenshot_path in enumerate(screenshots):
        screenshot_path = "own_dataset/" + screenshot_path
        #    screenshot_path = os.path.join(os.getcwd(), screenshot_path)  # Use current working directory
        try:
            img = Image.open(screenshot_path)
            tensor = transform(img).unsqueeze(0)  # Add batch dimension
            screenshot_tensors.append(tensor)
            valid_labels.append(labels[i])  # Keep the corresponding label only if the image exists
        except FileNotFoundError:
            print(f"File not found: {screenshot_path}, skipping.")
            continue

    if not screenshot_tensors:
        print("No valid screenshots found, skipping training.")
        return
    
    # Concatenate all tensors into one batch
    screenshot_tensors = torch.cat(screenshot_tensors).to(device)
    valid_labels = torch.tensor(valid_labels).float().to(device).unsqueeze(1)  # Reshape labels for training
    
    # Continue training
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classification_layer.parameters()), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(5):
        model.train()
        classification_layer.train()
        for i in range(0, len(screenshot_tensors), 32):  # Batch size 32
            inputs = screenshot_tensors[i:i+32]
            labels_batch = valid_labels[i:i+32]
            optimizer.zero_grad()
            embeddings = model(inputs)
            outputs = classification_layer(embeddings)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/5 completed.")

    model_path = "voting_screenshots.pth"
    # Save the updated model
    torch.save({
        'model_state_dict': model.state_dict(),
        'classification_layer_state_dict': classification_layer.state_dict()
    }, model_path)
    print("Screenshot model re-saved after additional training.")



# Function to generate predictions for screenshots using the ScreenshotModel and classification layer
def generate_screenshot_predictions(screenshots, model_path='voting_screenshots.pth', output_dim=256):
    model_path = "screenshots/screenshot_model_mixed.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained screenshot model and classification layer
    model = ScreenshotModel(output_dim=output_dim).to(device)
    classification_layer = torch.nn.Linear(output_dim, 1).to(device)

    # Load the saved model and classification layer weights
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    classification_layer.load_state_dict(checkpoint['classification_layer_state_dict'])

    model.eval()
    classification_layer.eval()

    # Define the transformation for image preprocessing (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet18 expects input size of 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ResNet
    ])

    predictions = []
    total_screenshots = len(screenshots)  # Get the total number of screenshots

    with torch.no_grad():
        for i, screenshot_path in enumerate(screenshots):
            screenshot_path = "own_dataset/" + screenshot_path

            try:
                # Load and preprocess the image
                image = Image.open(screenshot_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
                
                # Generate hidden representations from the screenshot model
                hidden_repr = model(image_tensor)
                
                # Use the classification layer to make predictions
                output = classification_layer(hidden_repr)
                
                # Apply sigmoid and get binary prediction
                pred = torch.sigmoid(output).squeeze().cpu().numpy()
                pred = (pred > 0.5).astype(int)  # Convert to binary prediction (0 or 1)
                predictions.append(pred)
            
            except FileNotFoundError as e:
                print(f"Error: File {screenshot_path} not found.")
                predictions.append(0)  # Default prediction for missing files

            # Progress print (percentage of screenshots processed)
            processed_percent = (i + 1) / total_screenshots * 100
            print(f"Processed {i + 1}/{total_screenshots} screenshots ({processed_percent:.2f}%)", end='\r')

    print("\nAll screenshots processed.")
    return predictions







from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def calculate_and_return_evaluation(true_labels, strict_predictions, lax_predictions):
    """
    Calculate and print Precision, Accuracy, F1, and Recall for both strict and lax predictions.
    Also outputs the confusion matrix for strict predictions.
    
    Args:
        true_labels (list or np.ndarray): The true labels of the dataset.
        strict_predictions (list or np.ndarray): The strict predictions from the voting mechanism.
        lax_predictions (list or np.ndarray): The lax predictions from the voting mechanism.
    """
    # Convert lists to arrays for easier computation (if they aren't already)
    true_labels = np.array(true_labels)
    strict_predictions = np.array(strict_predictions)
    lax_predictions = np.array(lax_predictions)

    # Strict prediction evaluation
    strict_accuracy = accuracy_score(true_labels, strict_predictions)
    strict_precision = precision_score(true_labels, strict_predictions)
    strict_recall = recall_score(true_labels, strict_predictions)
    strict_f1 = f1_score(true_labels, strict_predictions)

    # Lax prediction evaluation
    lax_accuracy = accuracy_score(true_labels, lax_predictions)
    lax_precision = precision_score(true_labels, lax_predictions)
    lax_recall = recall_score(true_labels, lax_predictions)
    lax_f1 = f1_score(true_labels, lax_predictions)

    # Print evaluation metrics for strict predictions
    print("Strict Voting Evaluation:")
    print(f"Accuracy: {strict_accuracy:.4f}")
    print(f"Precision: {strict_precision:.4f}")
    print(f"Recall: {strict_recall:.4f}")
    print(f"F1 Score: {strict_f1:.4f}")

    # Print evaluation metrics for lax predictions
    print("\nLax Voting Evaluation:")
    print(f"Accuracy: {lax_accuracy:.4f}")
    print(f"Precision: {lax_precision:.4f}")
    print(f"Recall: {lax_recall:.4f}")
    print(f"F1 Score: {lax_f1:.4f}")

    # Generate and display the confusion matrix for strict predictions
    cm = confusion_matrix(true_labels, strict_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Strict Voting Predictions")
    plt.show()