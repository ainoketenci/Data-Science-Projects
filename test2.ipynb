from io import TextIOWrapper
from PIL import Image
from zipfile import ZipFile 
import numpy as np

trnX = np.zeros((60000, 28, 28), dtype = "float32")
trnY = np.zeros((60000), dtype = "int32")
tstX = np.zeros((10000, 28, 28), dtype = "float32")
with ZipFile("ml410-20-sp-fashion.zip", "r") as archive:
    index = 0
    for i in range(trnX.shape[0]):
        with archive.open("fashion_trn/fashion_trn_" + str(i).zfill(5) + ".png") as file:
            img = Image.open(file)
            trnX[i] = np.asarray(img)
        index = index + 1
    with TextIOWrapper(archive.open("fashion_trn.csv", "r")) as file:
        header = file.readline()
        for i in range(trnY.shape[0]):
            trnY[i] = np.int32(file.readline().strip("\r\n").split(",")[1])
    index = 0
    for i in range(tstX.shape[0]):
        with archive.open("fashion_tst/fashion_tst_" + str(i).zfill(5) + ".png") as file:
            img = Image.open(file)
            tstX[i] = np.asarray(img)
        index = index + 1

trnX = trnX.reshape(trnX.shape[0], trnX.shape[1] * trnX.shape[2])
tstX = tstX.reshape(tstX.shape[0], tstX.shape[1] * tstX.shape[2])

# alternative normalization: ((pixelTensor / 255) - 0.5) * 2: [-1, 1]
trnX = trnX / 255
tstX = tstX / 255
