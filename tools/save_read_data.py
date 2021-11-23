import torch
import pickle

def save_obj(object, name):
    f = open(name+'.pkl', 'wb')
    pickle.dump(object, f)

def read_obj(name):
    f = open(name+'.pkl', 'rb')
    obj = pickle.load(f)
    return obj
