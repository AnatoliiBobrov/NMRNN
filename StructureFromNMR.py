import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader 
import torch.nn as nn
from datal import load_data, VECTOR_LENGTH, FEATURE_VECTOR_LENGTH, DTYPE, build_molecule, compare_molecules, DEVICE

EPOCHS = 10
VALIDATE_EACH_EPOCHS = 1
LENGTH = 20 #vector = v24
TRAINSET, TESTSET, VALIDSET = None, None, None
TRAIN_SET_LEN, VALID_LEN_SET, TEST_LEN_SET  = 0, 0, 0
train_size = 0.8
test_size = 0.5 # part of rest data
LR = 0.001
BATCH_SIZE = 2
TEST_MOOD = False
FILE_NAME = 'datav24_2_.pkl'

def pickle_db():
    print ("Start pickling database")
    database = load_data("nmr", length=-1, encoding='utf-8')
    with open('db.pkl', 'wb') as file:  
        pickle.dump(database, file)
    print ("Stop pickling database")
    return database

def load_sets():
    database = None
    try:
        with open('db.pkl', 'rb') as file:  
            database = pickle.load(file)
    except:
        database = pickle_db()  
        pass
    database = database[0:LENGTH]
    set_zero_seed()
    global TRAINSET, TESTSET, VALIDSET, TRAIN_SET_LEN, VALID_LEN_SET, TEST_LEN_SET
    TRAINSET, TESTSET = train_test_split(database, test_size=1-train_size)
    VALIDSET, TESTSET = train_test_split(TESTSET, test_size=test_size)
    TRAIN_SET_LEN = len(TRAINSET)
    VALID_LEN_SET = len(VALIDSET)
    TEST_LEN_SET = len(TESTSET)
    print (TRAIN_SET_LEN, VALID_LEN_SET, TEST_LEN_SET)

def set_zero_seed():
    np.random.seed(0)
    torch.manual_seed(0)

class Net1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = nn.Linear(VECTOR_LENGTH * 2, 2000, dtype=DTYPE)
        self.s2 = nn.ReLU()
        self.s3 = nn.Linear(2000, FEATURE_VECTOR_LENGTH, dtype=DTYPE)
        self.s4 = nn.Sigmoid() 

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        return x

class Userset(Dataset):  
    def __init__(self, data):  
        self.data = data  
    def __len__(self):  
        return len(self.data)  
    def __getitem__(self, idx):  
        return self.data[idx] 

def test_model(model, criterion, test_set_loader, test_len, device):
    begin = time.time()
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        for i in test_set_loader:
            labels = i[1].to(device)
            predicted = model(i[0].to(device))
            test_loss += criterion(predicted, labels).item()
            accuracy = compare_molecules(build_molecule(labels), build_molecule(predicted, True))
            test_accuracy += accuracy
    end = time.time()
    test_time = int(end-begin)
    return (test_accuracy/test_len, test_loss/test_len, test_time)

def train_and_test_batch():
    set_zero_seed()
    model = Net1()
    model.to(DEVICE)
    train_dset = Userset(TRAINSET)
    train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dset = Userset(VALIDSET)
    valid_loader = DataLoader(valid_dset, batch_size=1, shuffle=False)
    test_dset = Userset(TESTSET)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    outputs = None
    loss = None
    best_accuracy = 0
    for epoch in range(1, 1 + EPOCHS):
        begin = time.time()
        for i in train_loader:
            optimizer.zero_grad()
            outputs = model(i[0].to(DEVICE))
            loss = criterion(outputs, i[1].to(DEVICE))
            loss.backward()
            optimizer.step()
        if (epoch % VALIDATE_EACH_EPOCHS == 0):
            end = time.time()
            epoch_comp_time = int(end-begin)
            accuracy, loss, t_time = test_model(model, criterion, valid_loader, VALID_LEN_SET, DEVICE)           
            print(f"Epoch: {epoch:0{3}}, epoch time: {epoch_comp_time:0{4}}, valid time: {t_time:0{4}}, acc: {accuracy:.4f}, loss: {loss:.10f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                with open(FILE_NAME, 'wb') as file:  
                    pickle.dump(model, file)

    return load_and_test(FILE_NAME)
 
def load_and_test(name):
    test_dset = Userset(TESTSET)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False)
    set_zero_seed()
    model = None
    with open(name, 'rb') as file:  
        model = pickle.load(file) 
    model.to(DEVICE)
    criterion = torch.nn.MSELoss()
    accuracy, loss, t_time = test_model(model, criterion, test_loader, TEST_LEN_SET, DEVICE)           
    print(f"Test time: {t_time:0{4}}, accuracy: {accuracy:.4f}, loss: {loss:.10f}")
    return loss


if not TEST_MOOD:
    print ("Train mood")
    begin = time.time()
    load_sets()
    train_and_test_batch()
    end = time.time()
    print(f"Excecution time is {int(end-begin)}")
    input()
    
if TEST_MOOD:
    print ("Test mood")
    begin = time.time()
    load_sets()
    load_and_test(FILE_NAME)
    end = time.time()
    print(f"Excecution time is {int(end-begin)}")
    input()