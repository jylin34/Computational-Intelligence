import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化權重與偏置
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size)) # 產生一個input_size(3) x hidden_size(5) 的隨機[1,-1]矩陣
        self.bias_hidden = np.zeros(hidden_size) # 建立一個大小為hidden_size的零向量 numpy.ndarray
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_output = np.zeros(output_size)
        # 總共 3x5 + 5 + 5x3 + 3 = 38 個參數

    def forward(self, x): # x 是感測器的資料，一個 3 dimensional numpy.ndarray
        # 前向傳播
        self.hidden_layer = self.relu(np.dot(x, self.weights_input_hidden) + self.bias_hidden) # 一個 hidden_size(5) dimensional numpy.ndarray 向量
        self.output_layer = self.softmax(np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_output) # 一個 output_size(3) dimensional numpy.ndarray 向量
        return self.output_layer

    def sigmoid(self, x): # hidden layer
        x = np.clip(x, -500, 500)  # 限制輸入值的範圍
        return 1 / (1 + np.exp(-x))

    def softmax(self, x): # output layer
        exp_x = np.exp(x - np.max(x))  # 防止溢出
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def relu(self, x):
        return np.maximum(0, x) 

    def update_weights(self, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
        # 更新權重與偏置
        self.weights_input_hidden = weights_input_hidden
        self.bias_hidden = bias_hidden
        self.weights_hidden_output = weights_hidden_output
        self.bias_output = bias_output