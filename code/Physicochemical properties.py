import csv
import numpy as np
from propy.PyPro import GetProDes



def getMatrixLabelFingerprint(positive_position_file_name):

    rawseq11 = []

    with open(positive_position_file_name, 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:
            rawseq11.append(row[1])

    yyy = np.zeros(shape=(1547, 1))
    Matr = np.zeros((len(rawseq11), 1547))

    for index in range(0, len(rawseq11)):
        result = GetProDes(rawseq11[index]).GetALL()
        print(index)
        for p in range(0, 1547):
            yyy[p][0] = list(result.values())[p]
        for i in range(0, 1547):
            Matr[index][i] = yyy[i][0]

    return Matr

