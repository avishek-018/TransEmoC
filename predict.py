import argparse
import os
import ktrain
import numpy as np
from ktrain import text
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.metrics import precision_score,recall_score,f1_score
import logging
logger = logging.getLogger()
logger.disabled = True

def printTable(myDict, colList=None):

   if not colList: colList = list(myDict[0].keys() if myDict else [])
   myList = [colList] # 1st row = header
   for item in myDict: myList.append([str(item[col] if item[col] is not None else '') for col in colList])
   colSize = [max(map(len,col)) for col in zip(*myList)]
   formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
   myList.insert(1, ['-' * i for i in colSize]) # Seperating line
   for item in myList: print(formatStr.format(*item))



my_parser = argparse.ArgumentParser()

my_parser.add_argument('--sentence', action='store', type=str, required=True)
my_parser.add_argument('--model', action='store', type=str, default="model")
args = my_parser.parse_args()

test_data = [args.sentence]
model = args.model

model = ktrain.load_predictor(model)

classes = model.get_classes()
predictions = model.predict(test_data, return_proba=True)
predicted_class = model.predict(test_data)[0]

np.set_printoptions(suppress=True)
res = {classes[i]: round(predictions[0][i]*100, 4) for i in range(len(classes))}

print("Probabilty(%)\n")
printTable([res])
print("\nPredicted Class:", predicted_class)