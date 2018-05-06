import decisionTree, random
import csv


#load dataset
def processDataset(filename):
    dataset = []
    file = open(filename, 'rb')
    fileToRead = csv.reader(file)
    attribute = fileToRead.next()
    # iterate through rows of actual data
    for row in fileToRead:
        dataset.append(dict(zip(attribute, row)))

    return dataset


def test(dataset):
    prunedTest = []
    prunedTrain = []
    testAcc = []
    trainSetAcc = []
    data = processDataset(dataset)
    # random pick data form dataset and test, return the mean accuracy
    for i in range(35):
        random.shuffle(data)
        trainSet = data[:2 * len(data) / 3]
        testSet = data[2 * len(data) / 3:]
        tree = decisionTree.buildTree(trainSet, 'Class')
        accTrain = decisionTree.test(tree, trainSet)
        accTest = decisionTree.test(tree, testSet)
        prunedTree = decisionTree.pruneTree(tree, testSet)
        accPruneTestset = decisionTree.test(prunedTree, testSet)
        accPruneTrainset = decisionTree.test(prunedTree, trainSet)
        trainSetAcc.append(accTrain)
        testAcc.append(accTest)
        prunedTrain.append(accPruneTrainset)
        prunedTest.append(accPruneTestset)
    print
    print "Before the pruning,the trainSet accuracy : ", sum(trainSetAcc) / len(trainSetAcc)
    print "Before the pruning,the testSet accuracy  : ", sum(testAcc) / len(testAcc)
    print "After the pruning, the trainSet accuracy : ", sum(prunedTrain) / len(prunedTrain)
    print "After the pruning, the testSet accuracy  : ", sum(prunedTest) / len(prunedTest)


print "---------------test first  dataSet ----------------------"
test('vote.csv')
print
print "---------------test second dataSet ----------------------"
print
test('haberman.csv')
print
print "---------------test third  dataSet ----------------------"
test('balloons.csv')



