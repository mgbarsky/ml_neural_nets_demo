import numpy as np


class MLPerceptron:
    """ A Multi-Layer Perceptron"""

    def __init__(self, inputs, targets, nhidden, beta=1, momentum=0.9, outtype='logistic'):
        """ Constructor """
        # Set up network size
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype

        # Initialise network
        self.weights1 = (np.random.rand(self.nin + 1, self.nhidden) - 0.5) * 2 / np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden + 1, self.nout) - 0.5) * 2 / np.sqrt(self.nhidden)



    def train(self, inputs, targets, eta, niterations):
        """ Train the thing """
        # Add the inputs that match the bias node
        inputs_with_bias = np.concatenate((inputs, -np.ones((self.ndata, 1))), axis=1)
        change = range(self.ndata)

        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))

        for n in range(niterations):

            self.outputs = self.forward(inputs_with_bias)

            error = 0.5 * np.sum((self.outputs - targets) ** 2)
            if error == 0:
                break
            if (np.mod(n, 10) == 0):
                print("Iteration: ", n+1, " Error: ", error)

            # Different types of output neurons

            # if self.outtype == 'linear':
            deltao = (self.outputs - targets) / self.ndata

            if self.outtype == 'logistic':
                deltao = self.beta * (self.outputs - targets) * self.outputs * (1.0 - self.outputs)
            elif self.outtype == 'softmax':
                deltao = (self.outputs - targets) * (self.outputs * (-self.outputs) + self.outputs) / self.ndata
            else:
                print("error")


            deltah = self.hidden * self.beta * (1.0 - self.hidden) * (np.dot(deltao, np.transpose(self.weights2)))

            # backpropagation of error
            updatew1 = eta * (np.dot(np.transpose(inputs_with_bias), deltah[:, :-1])) + self.momentum * updatew1
            updatew2 = eta * (np.dot(np.transpose(self.hidden), deltao)) + self.momentum * updatew2
            self.weights1 -= updatew1
            self.weights2 -= updatew2

            cm, accuracy = self.confusion_matrix(inputs, targets)
            if accuracy == 1.0:
                break


    def forward(self, inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs, self.weights1);
        self.hidden = 1.0 / (1.0 + np.exp(-self.beta * self.hidden))
        self.hidden = np.concatenate((self.hidden, -np.ones((np.shape(inputs)[0], 1))), axis=1)

        outputs = np.dot(self.hidden, self.weights2);

        # Different types of output neurons
        if self.outtype == 'linear':
            return outputs
        elif self.outtype == 'logistic':
            return 1.0 / (1.0 + np.exp(-self.beta * outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs), axis=1) * np.ones((1, np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs)) / normalisers)
        else:
            print("error")

    def confusion_matrix(self, inputs, targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)
        outputs = self.forward(inputs)

        nclasses = np.shape(targets)[1]

        if nclasses == 1:
            nclasses = 2
            outputs = np.where(outputs > 0.5, 1, 0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0) * np.where(targets == j, 1, 0))


        return cm, np.trace(cm) / np.sum(cm)


if __name__ == "__main__":
    anddata = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    xordata = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

    p = MLPerceptron(anddata[:, 0:2], anddata[:, 2:3], 2)
    p.train(anddata[:, 0:2], anddata[:, 2:3], 0.25, 1001)
    cm, accuracy = p.confusion_matrix(anddata[:, 0:2], anddata[:, 2:3])
    print("Confusion matrix:")
    print(cm)
    print("Accuracy: ", accuracy)

    q = MLPerceptron(xordata[:, 0:2], xordata[:, 2:3], 2, outtype='logistic')
    q.train(xordata[:, 0:2], xordata[:, 2:3], 0.25, 5001)
    cm, accuracy = q.confusion_matrix(xordata[:, 0:2], xordata[:, 2:3])
    print("Confusion matrix:")
    print(cm)
    print("Accuracy: ",accuracy)