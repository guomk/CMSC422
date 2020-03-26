import runClassifier
import linear
import datasets
import mlGraphics
from matplotlib.pyplot import show

f = linear.LinearClassifier({'lossFunction': linear.HingeLoss(), 'lambda': 1, 'numIter': 1000, 'stepSize': 0.5})
runClassifier.trainTestSet(f, datasets.WineDataBinary)

# mlGraphics.plotLinearClassifier(f, datasets.WineDataBinary.X, datasets.WineDataBinary.Y)
# show()