import random 
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

## Start of helper functions ####
class Tree:
    def __init__(self):
        self.left = None
        self.right = None
        self.value = None
        self.leaf = False

def calc_majority(arr):
    ones = np.count_nonzero(arr)
    zeros = len(arr)-ones
    if(zeros>ones):
        return 0
    return 1

def sequence_generator():
    k = 14 
    noise = 6
    seq = []
    value = None
    
    r = random.randint(0,1)
    if (r<0.5):
        seq.append(0)
    else:
        seq.append(1)
   
    for i in range(k):
        r = random.random()
        top = seq[-1]
        if(r<=0.75):
            seq.append(top)
        else:
            seq.append(1-top)
   
    for i in range(noise):
        r = random.random()
        if (r<0.5):
            seq.append(0)
        else:
            seq.append(1)
   
    if(seq[0]==0):
        value = calc_majority(seq[1:8])
    else:
        value = calc_majority(seq[8:15])
    seq.append(value)
    seq = np.array(seq)
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

def dataset_generator(m):
    data = []
    # target = []
    for j in range(m):
        s = sequence_generator()
        data.append(s)
        # target.append(v)
    data = np.array(data)
    columns = ["x"+ str(i)  for i in range(21)]
    columns.append("y")
    columns = np.array(columns)
    df_data = pd.DataFrame(data= data ,columns= columns )
    return df_data

def makeTree(data):
    # ig = []
    max_ig = -1
    root = Tree()
    var_name = None
    for i in range(data.shape[1]-1):
        ig = information_gain(data["x"+str(i)].values,data["y"].values)
        # print(ig)
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

### End of helper functions ##
# treeRoot = Tree()
# treeRoot = makeTree(data, treeRoot)

# def inOrderTraversal(node):
#     if(node==None):
#         return
#     inOrderTraversal(node.left)
#     print(node.value)
#     inOrderTraversal(node.right)

def inorderTraversal(root):
    current = root
    temp = []
    inOrderList = []
    while True:
        if(current):
            temp.append(current)
            current = current.left
        else:
            if(len(temp)!=0):
                node = temp.pop()
                inOrderList.append(node.value)
                current = node.right
            else:
                return inOrderList



def postOrderTraversal(node):
    if(node==None):
        return
    postOrderTraversal(node.left)
    postOrderTraversal(node.right)
    print(node.value)

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


## Q2 Part 1 plot
# m = []
# error = []
# for i in range(10,10000, 20):
#     m.append(i)
#     train_data = dataset_generator(i)
#     tree = makeTree(train_data)
#     test_data = dataset_generator(i)
#     error.append(testDecisionTree(test_data, tree))
#     if(i%110==0):
#         print(i)

# plt.plot(m, error)
# plt.ylabel("Error")
# plt.xlabel("M")
# plt.show()

data2 = dataset_generator(10)

############ Q2 Part 2 ###################
# m = []
# noiseVarCounts = []
# noiseVarList = ['x15', 'x16', 'x17', 'x18', 'x19', 'x20'] 

# for i in range(10,1000,20):
#     avgNumNoise = 0
#     m.append(i)
#     for k in range(10):
#         count = 0
#         train_data = dataset_generator(i)
#         test_data = dataset_generator(i)
#         tree = makeTree(train_data)
#         elements = inorderTraversal(tree)
#         for x in elements:
#             if(x in noiseVarList):
#                 count+=1
#         avgNumNoise+=count
#     avgNumNoise = avgNumNoise*1.0 / 10
#     noiseVarCounts.append(avgNumNoise)   

# plt.plot(m,noiseVarCounts)
# plt.xlabel("m")
# plt.ylabel("Noise Variable Count")
# plt.show()
# plt.figure()


###### Q2) Part 3 a) #######

data = dataset_generator(10000)
train_data = data[:8000]
test_data = data[8000:]

def makeTreeByPruningDepth(data, desired_depth, current_depth):
    max_ig = -1
    root = Tree()
    var_name = None
    for i in range(data.shape[1]-1):
        ig = information_gain(data["x"+str(i)].values,data["y"].values)
        if(ig>=max_ig):
            max_ig = ig
            var_name = "x"+str(i)
    if(current_depth == desired_depth):
        node = Tree()
        node.value = calc_majority(data["y"].values)
        node.leaf = True
        return node

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
        root.left = makeTreeByPruningDepth(zeroData, desired_depth, current_depth+1)
    if(oneData.shape[1] !=0 ):
        root.right = makeTreeByPruningDepth(oneData, desired_depth, current_depth+1) 
    return root


# depthPrunedTree = makeTreeByPruningDepth(train_data, 2, 0)

# m = []
# error_train = []
# error_test = []
# for i in range(21):
#     m.append(i)
#     tree = makeTreeByPruningDepth(train_data, i, 0 )
#     error_test.append(testDecisionTree(test_data, tree))
#     error_train.append(testDecisionTree(train_data, tree))

# plt.plot(m, error_train)
# plt.ylabel("Error Train")
# plt.xlabel("d")
# plt.show()

# plt.figure()
# plt.plot(m, error_test)
# plt.ylabel("Error Test")
# plt.xlabel("d")
# plt.show()


########## Q2 Part 3 b) ####################
def makeTreeByPruningSamples(data, sample_size):
    max_ig = -1
    root = Tree()
    var_name = None
    for i in range(data.shape[1]-1):
        ig = information_gain(data["x"+str(i)].values,data["y"].values)
        if(ig>=max_ig):
            max_ig = ig
            var_name = "x"+str(i)
    if( data.shape[0]<= sample_size):
        node = Tree()
        node.value = calc_majority(data["y"].values)
        node.leaf = True
        return node

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
        root.left = makeTreeByPruningSamples(zeroData, sample_size)
    if(oneData.shape[1] !=0 ):
        root.right = makeTreeByPruningSamples(oneData, sample_size) 
    return root

# sampleSizePrunedTree = makeTreeByPruningSamples(train_data, 7000)

m = []
error_train = []
error_test = []
for i in range(10, 1000, 20):
    m.append(i)
    tree = makeTreeByPruningSamples(train_data, i)
    error_test.append(testDecisionTree(test_data, tree))
    error_train.append(testDecisionTree(train_data, tree))

plt.plot(m, error_train)
plt.ylabel("Error Train")
plt.xlabel("s")
plt.show()

plt.figure()
plt.plot(m, error_test)
plt.ylabel("Error Test")
plt.xlabel("s")
plt.show()


################## Q5) ###################
d = 8 ## using elbow method
optimizedDepthPrunedTree = makeTreeByPruningDepth(train_data, d, 0)

################## Q6) ###################
sample = 8 ## using elbow method
optimizedDepthPrunedTree = makeTreeByPruningDepth(train_data, d, 0)