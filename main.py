# main.py
# -------
# Carl Gao, Lewin Xue

from dtree import *
import sys

DATASET_SIZE = 100

class Globals:
    noisyFlag = False
    pruneFlag = False
    valSetSize = 0
    dataset = None



##Classify
#---------

def classify(decisionTree, example):
    return decisionTree.predict(example)

##Learn
#-------
def learn(dataset):
    learner = DecisionTreeLearner()
    learner.train( dataset)
    return learner.dt

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def parseArgs(args):
  """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
    parseArgs([ 'main.py', '-n', '-p', 5 ]) = { '-n':True, '-p':5 }"""
  args_map = {}
  curkey = None
  for i in xrange(1, len(args)):
    if args[i][0] == '-':
      args_map[args[i]] = True
      curkey = args[i]
    else:
      assert curkey
      args_map[curkey] = args[i]
      curkey = None
  return args_map

def validateInput(args):
    args_map = parseArgs(args)
    valSetSize = 0
    noisyFlag = False
    pruneFlag = False
    boostRounds = -1
    maxDepth = -1
    if '-n' in args_map:
      noisyFlag = True
    if '-p' in args_map:
      pruneFlag = True
      valSetSize = int(args_map['-p'])
    if '-d' in args_map:
      maxDepth = int(args_map['-d'])
    if '-b' in args_map:
      boostRounds = int(args_map['-b'])
    return [noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds]

# calculates 0-1 accuracy
def scoreOnGivenDataset(learner, validationSet):
    successCounter = 0
    for example in validationSet.examples:
        if learner.predict(example) == example.attrs[-1]:
            successCounter += 1
    print float(successCounter) / len(validationSet.examples)
    return float(successCounter) / len(validationSet.examples)

def main():
    arguments = validateInput(sys.argv)
    noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds = arguments
    print noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds

    # Read in the data file
    
    if noisyFlag:
        f = open("noisy.csv")
    else:
        f = open("data.csv")

    data = parse_csv(f.read(), " ")
    dataset = DataSet(data)
    
    # Copy the dataset so we have two copies of it
    examples = dataset.examples[:]
 
    dataset.examples.extend(examples)
    dataset.max_depth = maxDepth
    if boostRounds != -1:
      dataset.use_boosting = True
      dataset.num_rounds = boostRounds

    # ====================================
    # WRITE CODE FOR YOUR EXPERIMENTS HERE
    # ====================================

    learner = DecisionTreeLearner()
    scoreList = []

    random.shuffle(data)
    
    for i in range(0, DATASET_SIZE, DATASET_SIZE/10):
        trainingSet = DataSet(data[:i] + data[i+10:DATASET_SIZE], values=dataset.values)
        validationSet = DataSet(data[i:i+DATASET_SIZE/10])
        learner.train(trainingSet)
        scoreList.append(scoreOnGivenDataset(learner, validationSet))
    print "average:", sum(scoreList)/len(scoreList)
                


main()
