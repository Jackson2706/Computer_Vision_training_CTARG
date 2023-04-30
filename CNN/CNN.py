import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("photo.jpeg")
img = cv2.resize(img, (200,200))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
# plt.imshow(img_gray)
# print(img.shape)
np.random.seed(47)

class Conv2d:
    def __init__(self, input, numOfKernel = 8, kernelSize = 3, padding = 0, stride=1):
        self.input = np.pad(input, ((padding,padding), (padding, padding)), "constant")
        self.stride = stride
        self.kernel = np.random.randn(numOfKernel,kernelSize, kernelSize)
        self.results = np.zeros((int((self.input.shape[0] - self.kernel.shape[1])/self.stride + 1),
                                  int((self.input.shape[1] - self.kernel.shape[2])/self.stride +1),
                                  self.kernel.shape[0]))
    
    def getROI(self):
        for row in range(int((self.input.shape[0] - self.kernel.shape[1])/self.stride + 1)): 
            for col in range(int((self.input.shape[1] - self.kernel.shape[2])/self.stride +1)):
                roi = self.input[row*self.stride: row*self.stride + self.kernel.shape[1], 
                                 col*self.stride: col*self.stride + self.kernel.shape[2]]
                yield row, col, roi
    
    def operate(self):
        for layer in range(self.kernel.shape[0]):
            for row, col, roi in self.getROI():
                self.results[row,col, layer] = np.sum(roi * self.kernel[layer, :, :])
        return self.results

class ReLu:
    def __init__(self, input):
        self.input = input
        self.result = np.zeros((self.input.shape[0],
                                 self.input.shape[1],
                                 self.input.shape[2]))
    
    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    self.result[row, col, layer] = 0 if self.input[row, col, layer] < 0 \
                                                            else self.input[row, col, layer]
        return self.result

class LeakyReLu:
    def __init__(self, input):
        self.input = input
        self.result = np.zeros((self.input.shape[0],
                                 self.input.shape[1],
                                 self.input.shape[2]))
    
    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    self.result[row, col, layer] = 0.1*self.input[row, col, layer] if self.input[row, col, layer] < 0 \
                                                            else self.input[row, col, layer]
        return self.result

class Maxpooling:
    def __init__(self, input, poolingSize):
        self.input = input
        self.poolingSize = poolingSize
        self.result = np.zeros((int(self.input.shape[0]/self.poolingSize), int(self.input.shape[1]/self.poolingSize), self.input.shape[2]))

    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(int(self.input.shape[0]/self.poolingSize)):
                for col in range(int(self.input.shape[1]/self.poolingSize)):
                    self.result[row, col, layer] = np.max(self.input[row*self.poolingSize: row*self.poolingSize+self.poolingSize,
                                                              col*self.poolingSize: col*self.poolingSize+self.poolingSize,
                                                              layer])
        return self.result

if __name__ == "__main__":
    img_gray_conv2 = Conv2d(img_gray, 16, 3,0, 1).operate()
    img_gray_conv2_relu = ReLu(img_gray_conv2).operate()
    img_gray_conv2_leakyrelu = LeakyReLu(img_gray_conv2).operate()
    img_gray_conv2_leakyrelu_maxpooling = Maxpooling(img_gray_conv2_leakyrelu,2).operate()
    fig = plt.figure(figsize=(10,10))
    for i in range(16):
        plt.subplot(4,4, i + 1)
        plt.imshow(img_gray_conv2_leakyrelu_maxpooling[:,:, i], cmap="gray")

    plt.savefig("img_gray_conv2d_leakyrelu_maxpooling.jpg")
    plt.show()
