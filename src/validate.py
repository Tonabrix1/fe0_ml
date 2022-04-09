import cv2
import numpy as np

def read(filename):
    with open(filename, 'r') as f:
        flat = list(map(lambda x: float(x)*255, f.read().split()));
    flat = np.reshape(flat,(28,28))
    return flat

im = read("data.txt")
print(im)
cv2.imshow('data',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
