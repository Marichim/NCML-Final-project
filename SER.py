import os
import torch
import torchaudio
import librosa
import glob
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from transformers import Wav2Vec2ForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# Constants
FIXED_DURATION = 3.0  # seconds
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Dataset Class
class RAVDESSDataset(Dataset):
    def __init__(self, file_list, labels, processor):
        self.file_list = file_list
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]

        # Load audio
        waveform, sr = librosa.load(file_path, sr=16000)
        
        # if np.random.rand() < 0.2:  # 50% chance to shift
            # waveform = shift(waveform, 16000, shift_max=0.2, shift_direction='both')
        # if np.random.rand() < 0.3:  # 50% chance to add noise
            # waveform = noise(waveform, noise_factor=0.005)

        # Pad or truncate to fixed duration
        target_length = int(16000 * FIXED_DURATION)
        if len(waveform) < target_length:
            pad_length = target_length - len(waveform)
            waveform = np.concatenate([waveform, waveform[::-1][:pad_length]])
        else:
            waveform = waveform[:target_length]
 
        # Process audio
        # inputs = self.processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True, max_length=160000, truncation=True)                
        
        return inputs.input_values.squeeze(0), label
    
# Load file paths and labels
def load_data(data_directory):
    
    """
    loading dataset

    Parameters
    ----------
    save : boolean, save the data to disk as .npy

    """
   #  x, y = [], []
    file_paths = []
    labels = []
    for file in glob.glob(data_directory + "/Actor_*/*.wav"):

        file_name = os.path.basename(file)
        
        # get emotion label from the file name
        emotion = EMOTIONS[file_name.split("-")[2]]  

        if emotion:
            file_paths.append(file)
            labels.append(emotion)
            
        
    return file_paths, labels

def noise(data, noise_factor):
    
    """
    add random white noises to the audio

    Parameters
    ----------
    data : np.ndarray, audio time series
    noise_factor : float, the measure of noise to be added 

    """
    noise = np.random.randn(len(data)) 
    augmented_data = data + noise_factor * noise
    
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


def shift(data, sampling_rate, shift_max, shift_direction):
    
    """
    shift the spectogram in a direction
    
    Parameters
    ----------
    data : np.ndarray, audio time series
    sampling_rate : number > 0, sampling rate
    shift_max : float, maximum shift rate
    shift_direction : string, right/both
    
    """
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
        
    return augmented_data

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = 'ravdess'  
    file_paths, labels = load_data(data_dir)
    
    # Encode labels
    le = LabelEncoder()
    le.fit(labels)
    labels_encoded = le.transform(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(file_paths, labels_encoded, test_size=0.2, random_state=42)

    # Load Wav2vec2 processor and model
    model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=8, problem_type="single_label_classification").to(device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    # model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device) 
    

    # Create datasets and loaders
    train_dataset = RAVDESSDataset(X_train, y_train, processor)
    test_dataset = RAVDESSDataset(X_test, y_test, processor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize classifier
    # classifier = CNNClassifier(input_dim=768, num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.0003)
 
  
    # Training loop
    print('Start Training')
    for epoch in range(15):  # Adjust number of epochs as needed
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # with torch.no_grad():
                # features = model(inputs).last_hidden_state.transpose(1, 2)  # (batch_size, features, seq_len)
            # features = model(inputs).last_hidden_state  # shape: (batch, seq_len, 768)
            # features = torch.mean(features, dim=1)
            outputs = model(inputs).logits
            labels = labels.long()  
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Evaluation
    # classifier.eval()
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # features = model(inputs).last_hidden_state.transpose(1, 2)
            outputs = model(inputs).logits
            # _, predicted = torch.max(outputs, 1)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    print(f"Accuracy: {100 * correct / total}%")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
