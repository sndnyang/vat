from .fnn import FNN
from source import layers as L


class FNN_syn(FNN):

    def __init__(self):
        self.l1 = L.Linear(size=(100, 100))
        self.l2 = L.Linear(size=(100, 2))
        self.params = self.l1.params + self.l2.params

    def forward(self, inputs, train=True):
        h = inputs
        h = self.l1(h)
        h = L.relu(h)
        h = self.l2(h)
        return L.softmax(h)

    def forward_no_update_batch_stat(self, inputs, train=True):
        return self.forward(inputs, train)
