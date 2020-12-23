import random 
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

class Tree:
    def __init__(self):
        self.left = None
        self.right = None
        self.value = None
        self.leaf = False

## Q1 Part 1 and 2 #######


def generate_weights(k):
    weights = []
    denominator = 0
    for i in range(1,k):
        denominator += 0.9 ** (i+1)
    for i in range(1,k):
        w = 0.9 ** (i+1)/denominator
        weights.append(w)
    return np.array(weights)

def sequence_generator(k):
    seq = []
    value = None
    r = random.random()
    if (r<0.5):
        seq.append(0)
    else:
        seq.append(1)
    for i in range(1,k):
        r = random.random()
        top = seq[-1]
        if(r<=0.75):
            seq.append(top)
        else:
            seq.append(1-top)
    weights = generate_weights(k)
    seq = np.array(seq)
    if (np.dot(seq[1:],weights)>=0.5):
        value = seq[0]
    else:
        value = 1- seq[0]
    seq = np.append(seq,value)
    return seq

def entropy(vals):
    if(len(vals)==0):
        return 0
    zeros = len(np.where(vals==0)[0])
    ones = len(np.where(vals==1)[0])
    prob0 = zeros*1.0/len(vals)
    prob1 = ones*1.0/len(vals)
    e = 0
    if(prob0!=0): 
        e += -1*(prob0 * math.log(prob0,2) )
    if(prob1!=0):
        e+= -1* ( prob1 * math.log(prob1,2))
    return e

def information_gain(x,y):
    Hy = entropy(y)
    px0 = len(np.where(x==0)[0])*1.0/len(x)
    px1 = len(np.where(x==1)[0])*1.0/len(x)
    hyx0 = entropy(np.take(y,np.where(x==0)[0]))
    hyx1 = entropy(np.take(y,np.where(x==1)[0]))
    Hyx = px0*hyx0 + px1*hyx1
    ig = Hy - Hyx
    return ig

def dataset_generator(k,m):
    data = []
    for j in range(m):
        s = sequence_generator(k)
        data.append(s)
    data = np.array(data)
    columns = ["x"+ str(i)  for i in range(1,k+1)]
    columns.append("y")
    columns = np.array(columns)
    df_data = pd.DataFrame(data= data ,columns= columns )
    return df_data

###### Q1 part 3 ###################
data = dataset_generator(4,30)


def makeTree(data):
    max_ig = -1
    root = Tree()
    var_name = None
    for i in range(1,data.shape[1]):
        ig = information_gain(data["x"+str(i)].values,data["y"].values)
        if(ig>=max_ig):
            max_ig = ig
            var_name = "x"+str(i)
    if(max_ig==0):
        node = Tree()
        node.value = data["y"].values[0]
        node.leaf = True
        return node
    root.value = var_name
    root.leaf = False
    zeroData = data[data[var_name] ==0] 
    oneData = data[data[var_name] ==1]
    if(zeroData.shape[1] !=0):
        root.left = makeTree(zeroData)
    if(oneData.shape[1] !=0 ):
        root.right = makeTree(oneData) 
    return root

treeRoot = Tree()
treeRoot = makeTree(data)

def inOrderTraversal(node):
    if(node==None):
        return
    inOrderTraversal(node.left)
    print(node.value)
    inOrderTraversal(node.right)

# inOrderTraversal(treeRoot)

# print("--------------------x----------------------x---------------------")
def postOrderTraversal(node):
    if(node==None):
        return
    postOrderTraversal(node.left)
    postOrderTraversal(node.right)
    print(node.value)
# postOrderTraversal(treeRoot)


def testDecisionTree(data, treeRoot):
    target = data['y'].values
    prediction = []
    df = data.drop('y', axis =1 )
    for index, row in df.iterrows():
        tempTreeNode = treeRoot
        while(not tempTreeNode.leaf):
            var_name = tempTreeNode.value
            if(row[var_name]==0):
                tempTreeNode = tempTreeNode.left
            else:
                tempTreeNode = tempTreeNode.right
        prediction.append(tempTreeNode.value)
    error = 0
    for x,y in zip(target,prediction):
        error += abs(x-y)
    error = error*1.0/len(target)
    return error


## Part 4 Average Error Rate:


avg_error = 0
for i in range(1000):
    test_data = dataset_generator(4,30)
    avg_error += testDecisionTree(test_data, treeRoot)
print("Average Error Rate :", avg_error/1000)


# Part 5 plot
m = []
error = []
for i in range(1,2000, 10):
    m.append(i)
    train_data = dataset_generator(10,i)
    tree = makeTree(train_data)
    test_data = dataset_generator(10, i)
    error.append(testDecisionTree(test_data, tree))

plt.plot(m, error)
plt.ylabel("Error")
plt.xlabel("M")
plt.show()


######## Part 6 ##########

def ginny_impurity(data,x,y):
    if(data.shape[0]==0):
        return 0
    left = data[data[x] ==0]["y"].values
    right = data[data[x] ==1]["y"].values
    gi_left = 0
    gi_right = 0
    if(len(left)>0):
        zeros = len(np.where(left==0)[0])*1.0/len(left)
        ones = len(np.where(left==1)[0])*1.0/len(left)
        px0 = zeros/(ones+zeros) 
        px1 = ones/(ones+zeros)

        gi_left = 1 - (px0**2+px1**2)

    if(len(right)>0):
        zeros = len(np.where(right==0)[0])*1.0/len(right)
        ones = len(np.where(right==1)[0])*1.0/len(right)
        px0 = zeros/(ones+zeros) 
        px1 = ones/(ones+zeros)
        gi_right = 1 - (px0**2+px1**2)

    gi = len(left)*gi_left/(len(left)+len(right)) + len(right)*gi_right/(len(left)+len(right)) 

    return gi


def makeTreeFromGinnyImpurity(data):
    min_gi = 1
    root = Tree()
    var_name = None
    for i in range(1,data.shape[1]):
        gi = ginny_impurity(data, "x"+str(i) ,data["y"].values)
        if(gi<=min_gi):
            min_gi = gi
            var_name = "x"+str(i)
    if(min_gi==0):
        node = Tree()
        node.value = data["y"].values[0]
        node.leaf = True
        return node
    root.value = var_name
    root.leaf = False
    zeroData = data[data[var_name] ==0] 
    oneData = data[data[var_name] ==1]
    if(zeroData.shape[1] !=0):
        root.left = makeTreeFromGinnyImpurity(zeroData)
    if(oneData.shape[1] !=0 ):
        root.right = makeTreeFromGinnyImpurity(oneData) 
    return root

data = dataset_generator(4,30)
tree = makeTreeFromGinnyImpurity(data)


m = []
error = []
for i in range(1,2000, 10):
    m.append(i)
    train_data = dataset_generator(10,i)
    tree = makeTreeFromGinnyImpurity(train_data)
    test_data = dataset_generator(10, i)
    error.append(testDecisionTree(test_data, tree))

plt.plot(m, error)
plt.ylabel("Error")
plt.xlabel("M")
plt.show()
