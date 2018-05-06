import math
import copy
import operator


# define node class, using the node to build up the tree which will be helpful to prune the tree
class Node:

    def __init__(self):
        self.subsets = []
        self.label = None
        self.children = {}
        self.child_index = 0
        self.classifiers = None

    def isLeaf(self):
        return self.label in self.classifiers


# if the dataset has the missilng value then use this fuction to fiexed the missing data.
def fixeDataset(dataset):
    cleanExamples = list()
    attributesUniqueValues = {}
    # grab all the unique values for each attribute
    for attribute in dataset[0].keys():
        values = [example[attribute] for example in dataset]
        uniqueValues = list(set(values))
        if '?' in uniqueValues:
            uniqueValues.remove('?')
            attributesUniqueValues[attribute] = attributesUniqueValues

    # Scan over all the data, and filled missing values
    for example in dataset:
        for attribute, value in example.iteritems():
            if value == '?':
                example[attribute] = findMostCommonValue(dataset, attribute, example['Class'],
                                                         attributesUniqueValues[attribute])

    return dataset


# find the most commom value for the attribute
def findMostCommonValue(dataset, attribute, classification, values):
    data = {}
    maxCount = 0
    maxAttributeValue = None

    for example in dataset:
        if example['Class'] == classification:
            currValue = example.get(attribute)
            if currValue != '?':
                if currValue in data.keys():
                    data[currValue] += 1
                else:
                    data[currValue] = 1

    for attributeValue, count in data.iteritems():
        if count > maxCount:
            maxCount = count
            maxAttributeValue = attributeValue

    return maxAttributeValue


# calculate the Entropy
def calEntropy(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for i in dataSet:  # the the number of unique elements and their frequency
        currentLabel = i[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    Entropy = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        Entropy -= prob * math.log(prob, 2)
    return Entropy


# gcalculate its information gain
def calInformationGain(attribute, dataSet, entropy):
    values = [example[attribute] for example in dataSet]
    uniqueValues = list(set(values))
    classificationDatasets = [[] for i in range(0, len(uniqueValues))]

    for example in dataSet:
        attributeValue = example[attribute]
        for i in range(0, len(uniqueValues)):
            if attributeValue == uniqueValues[i]:
                classificationDatasets[i].append(example['Class'])
    weights = [float(len(classificationDatasets[i])) / float((len(dataSet))) for i in
               range(0, len(classificationDatasets))]

    for dataset in classificationDatasets:
        if len(dataset) == 0:
            return 0  # have noise

    entropies = [calEntropy(classificationDatasets[i]) for i in range(0, len(classificationDatasets))]
    infoGain = entropy
    for i, val in enumerate(entropies):
        infoGain -= weights[i] * val

    return infoGain


# loop through attributes, find attribute with highest information gain.
def chooseBestAttribute(dataset):
    attributes = dataset[0].keys()
    # removes key: Class as it is not an attribute
    attributes.remove('Class')
    bestAttribute = None
    maxGain = 0
    classificationData = [example['Class'] for example in dataset]
    totalEntropy = calEntropy(classificationData)
    # calculate each attribute info Gain
    for attribute in attributes:
        gain = calInformationGain(attribute, dataset, totalEntropy)
        if gain > maxGain:
            maxGain = gain
            bestAttribute = attribute
    return bestAttribute


# According to the attribute with higeset information gain to split the dataset
def splitDataset(examples, bestAttribute):
    values = [example[bestAttribute] for example in examples]
    uniqueValues = list(set(values))
    subsets = [[] for i in range(0, len(uniqueValues))]
    for example in examples:
        value = example[bestAttribute]
        for i in range(0, len(uniqueValues)):
            if value == uniqueValues[i]:
                subsets[i].append(example)
    return subsets


# if in the tree, there is no same class, used to stop building tree
def isSameClass(dataset):
    baseClass = dataset[0]['Class']
    for example in dataset:
        if example['Class'] != baseClass:
            return False
    return True


# useing majority vote to determin classfication of the leaf node
def majorityCount(classList):
    values = list(set([example['Class'] for example in classList]))
    classCount = {}
    for i in values:
        if i not in classCount.keys(): classCount[i] = 0
        classCount[i] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def buildTree(dataset, attribute):
    # before building the tree ,fixed missing data first
    dataset = fixeDataset(dataset)
    # create root node
    root = Node()
    root.classifiers = list(set([example['Class'] for example in dataset]))
    # check if all classications are the same
    if isSameClass(dataset):
        root.label = dataset[0]['Class']
        return root
    # if no more attributes to split on, set the label or root to default value
    elif len(dataset[0]) == 1:
        root.label = attribute
        return root
    else:
        bestAttribute = chooseBestAttribute(dataset)
        if bestAttribute is None:
            root.label = attribute
            return root
        root.label = bestAttribute
        root.subsets = splitDataset(dataset, bestAttribute)
        for subset in root.subsets:
            majoritySubsets = majorityCount(subset)
            if len(subset) == 0:
                leaf = Node()
                leaf.label = majoritySubsets
                if len(root.children) == 0:
                    root.children[0] = leaf
                else:
                    keys = root.children.keys()
                    numOfKeys = len(root.children) - 1
                    index = keys[numOfKeys] + 1
                    root.children[index] = leaf
            else:
                if len(root.children) == 0:
                    root.children[0] = buildTree(subset, majoritySubsets)
                else:
                    keys = root.children.keys()
                    numKeys = len(root.children) - 1
                    index = keys[numKeys] + 1
                    root.children[index] = buildTree(subset, majoritySubsets)

        return root


'''
    Impletement the REP algorithmn to prune the tree
'''


def pruneTree(node, dataset):
    nodeList = [node]
    visited = []
    root = node
    prune = True
    # Evaluate pruning each possible node and remove the one that most improves accuracy
    while prune and len(nodeList) > 0:
        curNode = nodeList[0]
        oldacc = test(root, dataset)
        if curNode.isLeaf() or curNode.label in visited:
            if curNode in visited:
                for key in curNode.children.keys():
                    if key in curNode.children and not curNode.children[key].label in visited:
                        nodeList.append(curNode.children[key])

            nodeList.pop(0)
            continue

        visited.append(curNode.label)
        scores = pruneScores(curNode, dataset)
        maxScoreLabel, newacc = maxPruneGain(scores)

        if newacc < oldacc:
            prune = False
            break
        # it is greater or equal, so we need to prune the node form the tree
        root = pruneNode(node, maxScoreLabel)
        nodeList = [root]

    return root


# calculate if prune the node,then the new accuracy about the new tree.
def pruneScores(node, dataset):
    scores = {}
    nodeList = [node]
    rootNode = node
    while len(nodeList) > 0:
        prunenode = nodeList[0]
        if prunenode.isLeaf():
            nodeList.pop(0)
            continue
        scores[prunenode.label] = test(pruneNode(copy.deepcopy(rootNode), prunenode.label), dataset)

        for key in prunenode.children.keys():
            if key in prunenode.children:
                nodeList.append(prunenode.children[key])
        nodeList.pop(0)

    return scores


# find the node with higest inceased accuracy
def maxPruneGain(scores):
    keys = scores.keys()
    maxGain = scores[keys[0]]
    key = keys[0]

    for i in keys:
        if scores[i] > maxGain:
            maxGain = scores[i]
            key = i

    return key, maxGain


# prune node
def pruneNode(node, pruneLabel):
    nodeList = [node]
    root = node
    while len(nodeList) > 0:
        curNode = nodeList[0]
        if curNode.label == pruneLabel:
            totalEntries = []
            for subset in curNode.subsets:
                for example in subset:
                    totalEntries.append(example['Class'])

            values = list(set(totalEntries))
            maxCount = totalEntries.count(values[0])
            majorityClass = values[0]
            # prune the node and assign that leaf the most common classification of examples associated with that node.
            for i in values:
                if totalEntries.count(i) > maxCount:
                    maxCount = totalEntries.count(i)
                    majorityClass = i
            curNode.label = majorityClass
            curNode.children = {}
            return root

        for key in curNode.children.keys():
            if key in curNode.children:
                nodeList.append(curNode.children[key])

        nodeList.pop(0)

    return root


# Takes in a trained tree and a test set,then calculate the accuracy.
def test(rootNode, dataset):
    count = 0
    for example in dataset:
        if example['Class'] == predict(rootNode, example):
            count += 1

    return float(count) / float(len(dataset))


# Takes in a tree and one example.Returns the predict Class value that the tree assigns to the example.
def predict(rootNode, example):
    i = 0
    while rootNode.children:
        attribute = rootNode.label
        i += 1
        if example[attribute] == rootNode.subsets[0][0][attribute]:
            rootNode = rootNode.children[0]
        else:
            rootNode = rootNode.children[1]

    return rootNode.label


