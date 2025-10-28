class tensor:
    def __init__(self, X):
        self.value = X
        self.shape = [len(X), len(X[0])]

class cnn:
    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), padding=0, bias=5):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        self.kernel =  tensor([[0, -1, 0],
                        [-1, 5, -1], 
                        [0, -1, 0]])
    
    def _dot_prod(self, matrix1, matrix2):
        res = 0
        for i in range(len(matrix1)):
            for j in range(len(matrix1[0])):
                res += matrix1[i][j]*matrix2[i][j]
        return res
    
    def conv(self, X):
        h_out, w_out = self.output_shape(X)
        output = [[0 for _ in range(w_out) ] for _ in range(h_out)]
        
        for i in range(h_out):
            for j in range(w_out):
                temp_x = []
                for ti in range(3):
                    t = []
                    for tj in range(3):
                        t.append(X.value[ti+i][tj+j])
                    temp_x.append(t)

                output[i][j] = self._dot_prod(temp_x, self.kernel.value) + self.bias

        return output

    def output_shape(self, X):
        image = X
        h,w = image.shape[-2],image.shape[-1]
        k_h, k_w = self.kernel.shape[-2],self.kernel.shape[-1]
        
        h_out = (h-k_h-2*self.padding)//self.stride[0] +1
        w_out = (w-k_w-2*self.padding)//self.stride[1] +1
        return h_out,w_out

X = tensor([[1, 2, 3, 4], 
     [5, 6, 7, 8], 
     [9, 10, 11, 12],
     [13, 14, 15, 16]])

model = cnn(in_channel=1, out_channel=1, kernel_size=3)
print(model.output_shape(X))
print(model.conv(X))