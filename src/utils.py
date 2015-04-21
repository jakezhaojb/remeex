'''
Useful utilities
'''

import pickle   

def pickleIt(pyName, outputName):
    output = open(outputName, 'wb')
    pickle.dump(pyName, output)
    output.close()

def pickleLoad(inputName):
    pk1_file = open(inputName, 'rb')
    pyObj = pickle.load(pk1_file)
    return pyObj
