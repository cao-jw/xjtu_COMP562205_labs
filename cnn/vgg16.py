from sequential import Sequential
import pylayer as L


class VGG16(Sequential):
    def __init__(self, input_channel=3, output_class=1000):
        super().__init__([
            # 224x224xinput_channel
            L.Conv2d(input_channel, 2, (3, 3), padding=1, stride=1),
            L.ReLU(),
            L.Conv2d(2, 2, (3, 3), padding=1, stride=1),
            L.ReLU(),
            # 224x224x64
            L.MaxPool2d((2, 2), padding=0, stride=2),
            # 112x112x64
            L.Conv2d(2, 4, (3, 3), padding=1, stride=1),
            L.ReLU(),
            L.Conv2d(4, 4, (3, 3), padding=1, stride=1),
            L.ReLU(),
            L.MaxPool2d((2, 2), padding=0, stride=2),
            # 56x56x128
            L.Conv2d(4, 8, (3, 3), padding=1, stride=1),
            # L.ReLU(),
            # L.Conv2d(8, 8, (3, 3), padding=1, stride=1),
            # L.ReLU(),
            # L.Conv2d(8, 8, (3, 3), padding=1, stride=1),
            L.ReLU(),
            L.MaxPool2d((2, 2), padding=0, stride=2),
            # 28x28x256
            L.Conv2d(8, 16, (3, 3), padding=1, stride=1),
            L.ReLU(),
            # L.Conv2d(16, 16, (3, 3), padding=1, stride=1),
            # L.ReLU(),
            # L.Conv2d(16, 16, (3, 3), padding=1, stride=1),
            # L.ReLU(),
            L.MaxPool2d((2, 2), padding=0, stride=2),
            # 14x14x512
            L.Conv2d(16, 16, (3, 3), padding=1, stride=1),
            L.ReLU(),
            # L.Conv2d(16, 16, (3, 3), padding=1, stride=1),
            # L.ReLU(),
            # L.Conv2d(16, 16, (3, 3), padding=1, stride=1),
            # L.ReLU(),
            L.MaxPool2d((2, 2), padding=0, stride=2),
            #7x7x512
            L.Flatten(),
            L.Linear(7*7*16, 32),
            L.ReLU(),
            # L.Linear(32, 32),
            # L.ReLU(),
            L.Linear(32, output_class),
            L.CrossEntropyLossWithSoftmax(),
        ])
