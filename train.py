import argparse
import os
import ktrain
import pandas as pd
from ktrain import text
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.metrics import precision_score,recall_score,f1_score

my_parser = argparse.ArgumentParser()

my_parser.add_argument('--train', action='store', type=str, required=True)
my_parser.add_argument('--validation', action='store', type=str, required=True)
my_parser.add_argument('--text_col', action='store', type=str, required=True)
my_parser.add_argument('--label_col', action='store', type=str, required=True)

my_parser.add_argument('--model', action='store', type=str, default="xlm-roberta-base")
my_parser.add_argument('--epochs', action='store', type=int, default=20)
my_parser.add_argument('--batch_size', action='store', type=int, default=12)
my_parser.add_argument('--lr', action='store', type=float, default=2e-5)
my_parser.add_argument('--root', action='store', type=str, default="")
my_parser.add_argument('--maxlen', action='store', type=int, default="70")

args = my_parser.parse_args()

text_col = args.text_col
label_col = args.label_col
model_name = args.model
epochs = args.epochs
lr = args.lr
root = args.root
maxlen = args.maxlen
batch_size = args.batch_size


tr_filename = args.train
val_filename = args.validation

_, tr_file_extension = os.path.splitext(tr_filename)
_, val_file_extension = os.path.splitext(val_filename)

if tr_file_extension == '.xlsx':
  raw_train_data = pd.read_excel(root+tr_filename)
elif tr_file_extension == '.csv':
  raw_train_data = pd.read_csv(root+tr_filename)
else:
  print("Please use .xlsx of .csv files")

if val_file_extension == '.xlsx':
  raw_val_data = pd.read_excel(root+val_filename)
elif val_file_extension == '.csv':
  raw_val_data = pd.read_csv(root+val_filename)
else:
  print("Please use .xlsx of .csv files")


#Creating training data
train_data = raw_train_data[text_col].tolist()
train_labels = raw_train_data[label_col].tolist()

#Creating validation data
val_data = raw_val_data[text_col].tolist()
val_labels = raw_val_data[label_col].tolist()

## Preparing the model
transformer_model = text.Transformer(model_name, maxlen=maxlen)

## Processing the data
train = transformer_model.preprocess_train(train_data, train_labels)
val = transformer_model.preprocess_test(val_data, val_labels)

model = transformer_model.get_classifier() 
learner = ktrain.get_learner(model, train_data=train, val_data=val, batch_size=batch_size)


learner.autofit(lr, checkpoint_folder='chk', epochs=epochs)
predictor = ktrain.get_predictor(learner.model, preproc=transformer_model).save('model')