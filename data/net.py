import numpy as np
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import copy

class MyDataset(Dataset):
    # for iterate the dataset
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

class torch_net(torch.nn.Module):
    def __init__(self, input_dim,hidden_dim,output_dim,train_data,test_data):
        super(torch_net, self).__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.train_dataset = MyDataset(train_data)
        self.test_dataset = MyDataset(test_data)
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, collate_fn=self.collate_fn, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64,collate_fn=self.collate_fn, shuffle=True)
        self.hidden = torch.nn.Linear(input_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, output_dim)

    def collate_fn(self,examples):
        inputs = torch.tensor([ex[:-1].astype(float) for ex in examples],dtype=torch.float32)
        targets = torch.tensor([ex[-1] for ex in examples],dtype=torch.long)
        return inputs, targets

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

    def train(self,epochs,learning_rate=0.01):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(),lr=learning_rate)
        for epoch in range(epochs):
            TP = 0
            loss_lst = []
            for i, (train_x, labels) in enumerate(self.train_loader):
                y_pred = self(train_x)

                loss = criterion(y_pred, labels)
                loss_lst.append(loss.item())

                y_hat = copy.copy(y_pred)
                TP += torch.sum(labels.flatten() == torch.argmax(y_hat, dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = TP.data.numpy() / len(self.train_data)
            print("epoch:", epoch, "loss:", np.mean(loss_lst), "acc: %.3f" % acc)

    def test(self):
        # TP = 0
        TP_train,TN_train,FP_train,FN_train = 0,0,0,0
        for i, (test_x, y) in enumerate(self.test_loader):
            with torch.no_grad():
                y_pred = self(test_x)
            y_hat = copy.copy(y_pred)
            # TP += torch.sum(labels.flatten() == torch.argmax(y_hat, dim=1))

            y_hat = torch.argmax(y_hat, dim=1)
            Add_train = y_hat + y
            Sub_train = y_hat - y
            TP_train += np.sum(np.where(Add_train == 2, 1, 0))
            TN_train += np.sum(np.where(Add_train == 0, 1, 0))
            FP_train += np.sum(np.where(Sub_train == 1, 1, 0))
            FN_train += np.sum(np.where(Sub_train == -1, 1, 0))

        precision = TP_train / (TP_train + FP_train)
        recall = TP_train / (TP_train + FN_train)
        F_score = 2 * precision * recall / (precision + recall)
        # print(set + " Accuracy: %.3f" % (np.sum((prediction == y) / float(m))))
        print(TP_train,TN_train,FP_train,FN_train)
        print("Precision is %.3f" % precision)
        print("Recall is %.3f" % recall)
        print("F-score is %.3f" % F_score)

        # acc = TP.data.numpy() / len(self.test_data)
        # print("acc:", round(acc, 4), f"TP: {TP} / {len(self.test_data)}")

    def draw(self):
        pass

class FNN_network():
    def __init__(self):
        pass

    def relu(self,Z):
        A = np.maximum(0, Z)
        assert (A.shape == Z.shape)
        cache = Z
        return A, cache

    def softmax(self,Z):
        Z = np.array(Z,dtype=np.float64)
        Z_shift = Z - np.max(Z, axis=0)
        A = np.exp(Z_shift) / np.sum(np.exp(Z_shift), axis=0)

        cache = Z_shift
        return A, cache

    def initialize_parameters(self,n_x, n_h, n_y):
        W1 = np.random.randn(n_h, n_x) * 0.01  # weight matrix随机初始化
        b1 = np.zeros((n_h, 1))  # bias vector零初始化
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        return parameters

    def linear_forward(self,A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def linear_activation_forward(self,A_prev, W, b, activation):
        if activation == "softmax":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.softmax(Z)
        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def compute_cost(self,AL, Y):
        m = Y.shape[1]
        cost = -(np.sum(Y * np.log(AL))) / float(m)
        # cost = np.squeeze(cost)
        # assert (cost.shape == ())

        return cost

    def linear_backward(self,dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / float(m)
        db = np.sum(dZ, axis=1, keepdims=True) / float(m)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def relu_backward(self,dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)

        dZ[Z <= 0] = 0
        return dZ

    def softmax_backward(self,Y, cache):
        Z = cache
        s = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        dZ = s - Y
        return dZ

    def linear_activation_backward(self,dA, cache, activation):
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        elif activation == "softmax":
            dZ = self.softmax_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def update_parameters(self,parameters, grads, learning_rate):
        L = len(parameters) // 2

        for l in range(1, L + 1):
            parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * grads['dW' + str(l)]
            parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * grads['db' + str(l)]

        return parameters

    def train(self,X, Y, layers_dims, learning_rate=0.05, num_iterations=15000, print_cost=True):
        grads = {}
        costs = []
        (n_x, n_h, n_y) = layers_dims

        parameters = self.initialize_parameters(n_x, n_h, n_y)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Loop (gradient descent)
        for i in range(0, num_iterations + 1):
            state = np.random.get_state()
            np.random.shuffle(X)
            np.random.set_state(state)
            np.random.shuffle(Y)

            batch_size = 10
            n = len(X)
            mini_batches_x = [X[k:k + batch_size] for k in range(0, n, batch_size)]
            mini_batches_y = [Y[k:k + batch_size] for k in range(0, n, batch_size)]
            for i in range(len(mini_batches_x)):
                train_x = mini_batches_x[i]
                train_y = mini_batches_y[i]

                A1, cache1 = self.linear_activation_forward(train_x, W1, b1, activation='relu')
                A2, cache2 = self.linear_activation_forward(A1, W2, b2, activation='softmax')

                cost = self.compute_cost(A2, train_y)

                dA1, dW2, db2 = self.linear_activation_backward(train_y, cache2, activation='softmax')
                dA0, dW1, db1 = self.linear_activation_backward(dA1, cache1, activation='relu')

                grads['dW1'] = dW1
                grads['db1'] = db1
                grads['dW2'] = dW2
                grads['db2'] = db2

                parameters = self.update_parameters(parameters, grads, learning_rate)

                W1 = parameters["W1"]
                b1 = parameters["b1"]
                W2 = parameters["W2"]
                b2 = parameters["b2"]

                print("loss ", cost)

            if print_cost and (i % 1000 == 0):
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
                costs.append(cost)

        return parameters

def predict(X, y, parameters, set,net):
    m = X.shape[1]

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Forward propagation
    A1, _ = net.linear_activation_forward(X, W1, b1, activation='relu')
    probs, _ = net.linear_activation_forward(A1, W2, b2, activation='softmax')

    # print(probs)
    # convert probas to 0-9 predictions
    prediction = np.argmax(probs, axis=0)
    prediction = prediction.reshape(1, -1)

    Add_train = prediction + y
    Sub_train = prediction - y
    TP_train = np.sum(np.where(Add_train == 2, 1, 0))
    TN_train = np.sum(np.where(Add_train == 0, 1, 0))
    FP_train = np.sum(np.where(Sub_train == 1, 1, 0))
    FN_train = np.sum(np.where(Sub_train == -1, 1, 0))

    precision = TP_train / (TP_train + FP_train)
    recall = TP_train / (TP_train + FN_train)
    F_score = 2*precision*recall/(precision+recall)
    # print(set + " Accuracy: %.3f" % (np.sum((prediction == y) / float(m))))
    print("Precision is %.3f" % precision)
    print("Recall is %.3f" % recall)
    print("F-score is %.3f" % F_score)

    return prediction

def one_hot(x,n_class):
    n = x.shape[0]
    a = np.zeros((n,n_class))
    for i in range(n):
        a[i,int(x[i])] = 1
    return a

class Bayes():
    def __init__(self,train_data):
        self.train_data = train_data
        self.p_label_p,self.p_label_n,self.attr_p,self.attr_n,self.label_p,self.label_n = self.train(train_data)
        # print(self.attr_p,self.attr_n)

    def train(self,data):
        data_length = len(data)
        label_p = 0  # 正例数量
        for i in range(data_length):
            if str(data[i, data.shape[1] - 1]) == "1":
                label_p += 1
        label_n = data_length - label_p
        attr_p = dict()
        attr_n = dict()
        for i in data:
            for index in range(data.shape[1] - 1):
                if str(i[data.shape[1] - 1]) == "1":
                    if i[index] in attr_p.keys():
                        attr_p[i[index]] += 1
                    else:
                        attr_p[i[index]] = 1
                else:
                    if i[index] in attr_n.keys():
                        attr_n[i[index]] += 1
                    else:
                        attr_n[i[index]] = 1
        p_label_p = label_p / data_length
        p_label_n = 1 - p_label_p
        for i in attr_p.keys():
            attr_p[i] /= label_p
        for i in attr_n.keys():
            attr_n[i] /= label_n
        return p_label_p, p_label_n, attr_p, attr_n, label_p, label_n

    def predict(self,data):
        predict_label = []  # designed for evaluate
        target_label = []
        for i in data:
            predict_p = 1
            predict_n = 1
            for index in range(data.shape[1] - 1):
                if i[index] in self.attr_p.keys():
                    predict_p *= self.attr_p[i[index]]
                else:
                    predict_p *= 1 / self.label_p
                if i[index] in self.attr_n.keys():
                    predict_n *= self.attr_n[i[index]]
                else:
                    predict_n *= 1 / self.label_n

            predict_p *= self.p_label_p
            predict_n *= self.p_label_n
            if predict_p >= predict_n:
                predict_label.append(1)
            else:
                predict_label.append(0)
            if str(i[data.shape[1] - 1]) == "1":
                target_label.append(1)
            else:
                target_label.append(0)
        return predict_label, target_label

    def evaluate(self,predict_label,target_label):
        length = len(predict_label)
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        for i in range(length):
            if predict_label[i] == target_label[i]:
                if predict_label[i] == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if target_label[i] == 1:
                    FN += 1
                else:
                    FP += 1
        acc = (TP + TN) / (TP + TN + FP + FN)
        recall = TP / (TP + FN)
        precision = TP/(TP+FP)
        F_score = 2*precision*recall/(precision+recall)
        print(TP,FN,FP,TN)
        print("Precision is %.3f" % precision)
        print("Recall is %.3f" % recall)
        print("F-score is %.3f" % F_score)