import os
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import codecs
from transformers import GPT2tknizer, GPT2LMHeadModel
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm.auto import tqdm
import locale
import os
from rouge_score import rouge_scorer


os.environ['HF_TOKEN'] = "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"

path = 'Reviews.csv'
data = pd.read_csv(path)
data = data.sample(n=10000, random_state=42)
data.head()


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to clean text
def clean_text(text):
    #This fucntion deals with cleaning the text , which involves 
    #removing html tags 
    #converting the text into lower case 
    #removing punctuations and txts
    #Tokenizing the text 
    #Removing stop words 
    #lemmatizing
    #finally joining the words back together
    # Check if the text is a string
    if not isinstance(text, str):
        return "" 
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(clean_tokens)

data['Text'] = data['Text'].apply(clean_text)
data['summ'] = data['summ'].apply(clean_text)

#GPT - 2
#including the token path to access the text file 
token_path = 'hf_token.txt'
#Reading the token from the file with UTF-8 encoding and remove any BOM
with codecs.open(token_path, 'r', encoding='utf-8-sig') as file:
    hf_token = file.read().strip()
print("Token loaded successfully.")

# Ensuring the tknizer has the padding token set
tknizer = GPT2tknizer.from_pretrained('gpt2', token=hf_token)
tknizer.pad_token = tknizer.eos_token  

model = GPT2LMHeadModel.from_pretrained('gpt2', token=hf_token)
# Now we split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

# checking the shape of the training and testing set
print(f"Training set size: {train_data.shape[0]} entries")
print(f"Testing set size: {test_data.shape[0]} entries")

#We are encoding the text and summ using the hugging tknizer
# Using the tknizer's batch_encode_plus method to handle both text and summ simultaneously
#Now we are extracting the input ids and attention mask from the encoded pair
# The shape is of the format =  [batch_size, num_tokens]
# We are using inputid as labels for training in tasks like summarization

class RevDataset(Dataset):
    def __init__(self, dataframe, tknizer, maxLen=512):
        self.tknizer = tknizer
        self.txts = dataframe['Text'].tolist()
        self.summaries = dataframe['summ'].tolist()
        self.maxLen = maxLen

    def __len__(self):
        return len(self.txts)

    def __getitem__(self, idx):
        text = self.txts[idx]
        summ = self.summaries[idx]

       
        encodePair = self.tknizer.encode_plus(
            text, summ,
            add_special_tokens=True,
            maxLen=self.maxLen,
            truncation=True,
            padding='maxLen',
            return_tensors='pt'
        )

        
        inputid = encodePair['inputid'].squeeze(0)  
        attenMask = encodePair['attenMask'].squeeze(0)

        return {
            'inputid': inputid,
            'attenMask': attenMask,
            'labels': inputid.clone() 
        }
# Assume 'tknizer' is already defined (e.g., GPT2tknizer from Hugging Face)
train_dataset = RevDataset(train_data, tknizer)
test_dataset = RevDataset(test_data, tknizer)


# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
batch_size = 2
epochs = 3
learning_rate = 5e-5

# opti
opti = AdamW(model.parameters(), lr=learning_rate)

trainLoad = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testLoad = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model.train()  # Set the model to training mode

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_loss = 0
    for batch in tqdm(trainLoad):
        # Move batch to device
        inputid = batch['inputid'].to(device)
        attenMask = batch['attenMask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(inputid, attenMask=attenMask, labels=labels)
        loss = outputs.loss

        # Backward pass
        opti.zero_grad()
        loss.backward()
        opti.step()

        epoch_loss += loss.item()

    print(f"Average loss: {epoch_loss / len(trainLoad)}")


# Define the path to save the model
modelPath = '/content/drive/MyDrive/IR/ass4/gpt2_2_inshallah'

# Create the directory if it does not exist
if not os.path.exists(modelPath):
    os.makedirs(modelPath)
    print(f"Directory {modelPath} created")
else:
    print(f"Directory {modelPath} already exists")


# Setting the model to evaluation mode
model.save_pretrained(modelPath)
tknizer.save_pretrained(modelPath)
print("Model and tknizer have been saved successfully.")

model.eval()  
with torch.no_grad():
    totLoss = 0
    for batch in testLoad:
        inputid = batch['inputid'].to(device)
        attenMask = batch['attenMask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(inputid, attenMask=attenMask, labels=labels)
        loss = outputs.loss
        totLoss += loss.item()

    print(f"Validation Loss: {totLoss / len(testLoad)}")



# Trying to set locale to 'en_US.UTF-8'
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    os.environ['LC_ALL'] = 'en_US.UTF-8'


# Loading the pre-trained model and tokenizer
model_path = '/content/drive/MyDrive/IR/ass4/gpt2_2_shikha'
tknizer = GPT2tknizer.from_pretrained(model_path)
tknizer.padding_side = "left"  # Ensure padding is from left
model = GPT2LMHeadModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def summaryGen(review, maxLen=100):
    inputid = tknizer.encode(review, return_tensors='pt').to(device)

    generated = inputid  

    for _ in range(maxLen):
        outputs = model(generated)
        logits = outputs.logits

        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, axis=-1).unsqueeze(-1)

        generated = torch.cat([generated, next_token_id], dim=-1)

        if next_token_id == tknizer.eos_token_id:
            break

    summ = tknizer.decode(generated[0], skip_special_tokens=True)
    return summ

def metric(reference, generated):
    reference_words = set(reference.split())
    generated_words = set(generated.split())

    # Calculating the metrics : 
    #precision 
    #recall
    #F-score
    truePos = len(generated_words & reference_words)
    precision = truePos / len(generated_words) if generated_words else 0
    recall = truePos / len(reference_words) if reference_words else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

#Taking user input for the review text and summary 
review_text = input("Enter review text: ")
actual_summ = input("Enter actual summ: ")

# Generating the summary using the pretrained model
generated_summ = summaryGen(review_text)
print("Generated summ:", generated_summ)

# Computing the  metrics based on the actual and generated summary
metrics = metric(actual_summ, generated_summ)
print("Metrics (Simulated ROUGE-like Scores):")
print(f"Precision: {metrics['precision']:.2f}")
print(f"Recall: {metrics['recall']:.2f}")
print(f"F1-Score: {metrics['f1_score']:.2f}")


# Loading the fine-tuned model and toknizer
model_path = 'gpt2_2_shikha'
tknizer = GPT2tknizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def summaryGen(review, maxLen=100):
    inputid = tknizer.encode(review, return_tensors='pt').to(device)

    generated = inputid 

    for _ in range(maxLen):
        outputs = model(generated)
        logits = outputs.logits

        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, axis=-1).unsqueeze(-1)

        generated = torch.cat([generated, next_token_id], dim=-1)

        if next_token_id == tknizer.eos_token_id:
            break

    summ = tknizer.decode(generated[0], skip_special_tokens=True)
    return summ

def rogueCalc(generated, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores


review = "great taffy great price wide assortment yummy taffy delivery quick taffy lover deal"
reference_summ = "Great taffy at a great price. Wide assortment, quick delivery."
generated_summ = summaryGen(review)

# Calculating the rouge scores
rouge_scores = rogueCalc(generated_summ, reference_summ)
print("Generated summ:", generated_summ)
print("ROUGE-1: Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}".format(
    rouge_scores['rouge1'].precision, rouge_scores['rouge1'].recall, rouge_scores['rouge1'].fmeasure))
print("ROUGE-2: Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}".format(
    rouge_scores['rouge2'].precision, rouge_scores['rouge2'].recall, rouge_scores['rouge2'].fmeasure))
print("ROUGE-L: Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}".format(
    rouge_scores['rougeL'].precision, rouge_scores['rougeL'].recall, rouge_scores['rougeL'].fmeasure))