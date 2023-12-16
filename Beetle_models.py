from transformers import AutoTokenizer
import os
from xml.etree import ElementTree
from torch.utils.data import Dataset
import torch
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
import numpy as np

def compute_metrics(pred):
    labels = pred.label_ids
    print(pred.predictions)
    #preds = np.asarray(pred.predictions[0 ]).argmax(-1)
    preds = np.asarray(pred.predictions).argmax (-1)
    f1_micro = f1_score(labels,preds,average='micro')
    f1_macro = f1_score(labels,preds, average='macro')
    precision = precision_score(labels,preds,average='weighted')
    recall = recall_score(labels,preds,average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall
    }


def pull_training_dataset():
    training_variable = []
    target_variable = []

    path = r"C:\Users\Ufuk Cetinkaya\PycharmProjects\Huggingface\ThreeWayData_beetle" # dataset directory
    all_file_names = os.listdir(path)
    for x in range(len(all_file_names)):
        file_name = all_file_names[x]
        full_file = os.path.abspath(os.path.join('ThreeWayData_beetle', file_name))
        dom = ElementTree.parse(full_file)
        reference_answer = dom.findall('referenceAnswers/referenceAnswer')
        best_reference_answer = []  # alle besten Antworten
        for idx, val in enumerate(reference_answer):
            if (reference_answer[idx].attrib['category'] == "BEST"):
                best_reference_answer.append(val.text)
                                                                        #------------------ hol alle reference Antworten( nur die besten Antworten)
        student_answer = dom.findall('studentAnswers/studentAnswer')
        for x in range(len(best_reference_answer)):
            for y in range(len(student_answer)):
                training_variable.append(best_reference_answer[x] + "[SEP]" + student_answer[y].text)
                if (student_answer[y].attrib['accuracy'] == "correct"):
                    target_variable.append(0)
                elif (student_answer[y].attrib['accuracy'] == "incorrect"):
                    target_variable.append(1)
                else:
                    target_variable.append(2)
                                                                    #--------------------- alle reference Antworten mit allen studenten Antworten
    return training_variable, target_variable                       #--------------------- target Variablen : korrekt -> 0 und inkorrekt -> 1




def pull_test_dataset():

    test_unseen_answers_variable = []
    test_unseen_answers_target_variable = []

    test_unseen_question_variable = []
    test_unseen_question_target_variable = []


    path_1 = r"C:\Users\Ufuk Cetinkaya\PycharmProjects\Huggingface\test\3way\beetle\test-unseen-answers"  # testset directory
    path_2 = r"C:\Users\Ufuk Cetinkaya\PycharmProjects\Huggingface\test\3way\beetle\test-unseen-questions"  # testset directory
    all_test_file_names_1 = os.listdir(path_1)
    all_test_file_names_2 = os.listdir(path_2)

    for x in range(len(all_test_file_names_1)):
        file_name = all_test_file_names_1[x]
        full_file = os.path.abspath(os.path.join('test/3way/beetle/test-unseen-answers' ,file_name))
        dom = ElementTree.parse(full_file)
        reference_answer = dom.findall('referenceAnswers/referenceAnswer')
        best_reference_answer = []  # alle besten Antworten
        for idx, val in enumerate(reference_answer):
            if (reference_answer[idx].attrib['category'] == "BEST"):
                best_reference_answer.append(val.text)

        student_answer = dom.findall('studentAnswers/studentAnswer')
        for x in range(len(best_reference_answer)):
            for y in range(len(student_answer)):
                test_unseen_answers_variable.append(best_reference_answer[x] + " [SEP] " + student_answer[y].text)
                if (student_answer[y].attrib['accuracy'] == "correct"):
                    test_unseen_answers_target_variable.append(0)
                elif (student_answer[y].attrib['accuracy'] == "incorrect"):
                    test_unseen_answers_target_variable.append(1)
                else:
                    test_unseen_answers_target_variable.append(2)


    for x in range(len(all_test_file_names_2)):
        file_name = all_test_file_names_2[x]
        full_file = os.path.abspath(os.path.join('test/3way/beetle/test-unseen-questions', file_name))
        dom = ElementTree.parse(full_file)
        reference_answer = dom.findall('referenceAnswers/referenceAnswer')
        best_reference_answer = []  # alle besten Antworten
        for idx, val in enumerate(reference_answer):
            if (reference_answer[idx].attrib['category'] == "BEST"):
                best_reference_answer.append(val.text)

        student_answer = dom.findall('studentAnswers/studentAnswer')
        for x in range(len(best_reference_answer)):
            for y in range(len(student_answer)):
                test_unseen_question_variable.append(best_reference_answer[x] + " [SEP] " + student_answer[y].text)
                if (student_answer[y].attrib['accuracy'] == "correct"):
                    test_unseen_question_target_variable.append(0)
                elif (student_answer[y].attrib['accuracy'] == "incorrect"):
                    test_unseen_question_target_variable.append(1)
                else:
                    test_unseen_question_target_variable.append(2)

    return  test_unseen_answers_variable, test_unseen_answers_target_variable, test_unseen_question_variable, test_unseen_question_target_variable



class ThreeWayDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


model_name = "microsoft/deberta-v2-xlarge-mnli"  # Name vom BaseModel
model = AutoModelForSequenceClassification.from_pretrained(model_name)  # Base Model initialisiert


training_input,training_target = pull_training_dataset()                             # Trainings-Daten

test_unseen_answer_input,test_unseen_answer_target ,test_unseen_questions_input, test_unseen_questions_target = pull_test_dataset()# Test-Daten

tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False)                       # Tokenizer vom Base-model


train_encodings = tokenizer(training_input,truncation=True, padding=True)   # trainingsdate tokenized


test_unseen_answers_encodings = tokenizer(test_unseen_answer_input,truncation=True, padding=True)   # trainingsdate tokenized
test_unseen_questions_encodings = tokenizer(test_unseen_questions_input,truncation=True, padding=True)   # trainingsdate tokenized

train_dataset = ThreeWayDataset(train_encodings,training_target)            # auf pytorch dataset konfiguriert
test_unseen_answers_dataset = ThreeWayDataset(test_unseen_answers_encodings,test_unseen_answer_target)
test_unseen_questions_dataset = ThreeWayDataset(test_unseen_questions_encodings,test_unseen_questions_target)


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=24,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    warmup_steps=1024,
    #resume_from_checkpoint="./results/checkpoint-45500",
    fp16=True,
    logging_dir='./logs'
)

trainer = Trainer(
    model= model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_unseen_questions_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics

)

trainer.train()





#trainer.train("./results/checkpoint-65500")
#trainer.train(resume_from_checkpoint=True) # nimmt das letzte checkpoint
output = trainer.predict(test_unseen_answers_dataset)
print(output)

