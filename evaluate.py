import argparse
import os
import ktrain
import pandas as pd
from ktrain import text
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.metrics import precision_score,recall_score,f1_score

my_parser = argparse.ArgumentParser()

my_parser.add_argument('--test', action='store', type=str, required=True)
my_parser.add_argument('--model', action='store', type=str, default="model")
my_parser.add_argument('--text_col', action='store', type=str, required=True)
my_parser.add_argument('--label_col', action='store', type=str, required=True)

args = my_parser.parse_args()

test_path = args.test
model = args.model
text_col = args.text_col
label_col = args.label_col

model = ktrain.load_predictor(model)

_, file_extension = os.path.splitext(test_path)

if file_extension == '.xlsx':
  raw_test_data = pd.read_excel(test_path)
elif file_extension == '.csv':
  raw_test_data = pd.read_csv(test_path)
else:
  print("Please use .xlsx of .csv files")


#Creating training data
test_data = raw_test_data[text_col].tolist()
test_labels = raw_test_data[label_col].tolist()

y_pred = model.predict(test_data)
true, pred = test_labels, y_pred
print(confusion_matrix(true,pred))
print(classification_report(true,pred, target_names=model.get_classes()))
print("\nPrecison : ",precision_score(true, pred, average = 'weighted'))
print("\nRecall : ",recall_score(true, pred,  average = 'weighted'))
print("\nF1 : ",f1_score(true, pred,  average = 'weighted'))


with open("result.txt", 'a') as f:
  f.write("Confusion Marix:\n")
  f.write(str(confusion_matrix(true,pred)))
  f.write("\n")
  f.write(str(classification_report(true,pred, target_names=model.get_classes())))
  f.write("\n")
  f.write("\nPrecison : "+str(precision_score(true, pred, average = 'weighted')))
  f.write("\n")
  f.write("\nRecall : "+str(recall_score(true, pred,  average = 'weighted')))
  f.write("\n")
  f.write("\nF1 : "+str(f1_score(true, pred,  average = 'weighted')))


