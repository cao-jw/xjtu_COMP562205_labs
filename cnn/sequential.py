import pylayer as L
import cupy as cp

class Sequential(object):
    '''
        The sequential model is a list of layers in it's order,
        the __init__() method takes a list of layers.

        A layer must have forward() and backward() method,
        the backward() method of each layer returns
            - grad_input
            - grad_weight (optional, or other names)
            - grad_bias (optional, or other names)

        For flexible implementations of the optimizers, the sum of param grads
        will be stored and accumulated after calling Sequential.backward().

        The last layer must be criterion (CrossEntropyLossWithSoftmax).
    '''
    def __init__(self, layers):
        self.layers = layers
        self.param_grads = None # will be a list of tuples containing tensors
    
    def forward(self, input, gt_label, train_mode=True):
        x = input
        # forward the layers except the last one
        for l in self.layers[:-1]:
            # print(x.shape)
            # print(type(l))
            if (type(l) in (L.BatchNorm1d, L.BatchNorm2d, L.BottleNeck, L.BasicBlock)):
                x = l.forward(x, train_mode)
            else:
                x = l.forward(x)
        # the last one is criterion (CrossEntropyLossWithSoftmax)
        loss = self.layers[-1].forward(x, gt_label)
        return self.layers[-1].prob, loss

    def backward(self, grad):
        grads = []
        for l in reversed(self.layers):
            # print(grad.shape)
            # print(type(l))
            bwd_ret = l.backward(grad)
            if (isinstance(bwd_ret, tuple)):
                grad = bwd_ret[0]
                if (len(bwd_ret) > 1): # have trainable params
                    grads.append(bwd_ret[1:])
            else: # only grad_input
                grad = bwd_ret
                grads.append(())
        grads.reverse()
        self.param_grads = grads
