import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
import ast 
from utils.preprocessing import PubMedDataset
import torch

# load the dataset
df = pd.read_csv('../data/sampled_data.csv')
df["meshroot"] = df["meshroot"].apply(ast.literal_eval)  # apply literal_eval because lists appear as strings

# Assume the 'mushroot' column contains the labels and 'text' contains the article text
texts = df['abstractText'].tolist()
labels = df['meshroot'] 

# Use MultiLabelBinarizer to binarize the labels
mlb = MultiLabelBinarizer()
binarized_labels = mlb.fit_transform(labels)

# Split the dataset into train, validation, and test sets
texts_train, texts_temp, labels_train, labels_temp = train_test_split(texts, binarized_labels, test_size=0.3, random_state=42)
texts_val, texts_test, labels_val, labels_test = train_test_split(texts_temp, labels_temp, test_size=0.5, random_state=42)

# Tokenize the texts
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings_train = tokenizer(texts_train, truncation=True, padding=True, max_length=512)
encodings_val = tokenizer(texts_val, truncation=True, padding=True, max_length=512)
encodings_test = tokenizer(texts_test, truncation=True, padding=True, max_length=512)


train_dataset = PubMedDataset(encodings_train, labels_train)
val_dataset = PubMedDataset(encodings_val, labels_val)
test_dataset = PubMedDataset(encodings_test, labels_test)

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(mlb.classes_))

# Freeze all the layers except the classifier head
for param in model.bert.parameters():
    param.requires_grad = False

# Apply LoRA
config = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.1)
model = get_peft_model(model, config)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=120,
    per_device_eval_batch_size=120,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()


# Save the model and tokenizer
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')

# Load the model and tokenizer for predictions
loaded_model = BertForSequenceClassification.from_pretrained('./saved_model')
loaded_tokenizer = BertTokenizer.from_pretrained('./saved_model')

# Make predictions on new text
loaded_model.eval()
new_text = "This is a new medical article abstract."
inputs = loaded_tokenizer(new_text, return_tensors='pt', truncation=True, padding=True, max_length=512)

with torch.no_grad():
    outputs = loaded_model(**inputs)
    logits = outputs.logits

probs = torch.sigmoid(logits).cpu().numpy()
threshold = 0.5
predicted_labels = (probs > threshold).astype(int)
predicted_label_names = mlb.inverse_transform(predicted_labels)

print("Predicted labels:", predicted_label_names)
