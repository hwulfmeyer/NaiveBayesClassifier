"""
This file is for executing everything together
"""
import filehandling as fiha
import naivebayes as naba


filepathnames = "datasets/cardaten/carnames.data"
filepathdata = "datasets/cardaten/car.data"

# filepathnames = "datasets/tennisdaten/tennisnames.data"
# filepathdata = "datasets/tennisdaten/tennis.data"

# read data
classes, attributes, attribute_values = fiha.read_data_names(filepath=filepathnames)
instances = fiha.read_data(filepath=filepathdata)
print("Number of Classes: " + str(len(classes)))
print("Number of Attributes: " + str(len(attributes)))
train_data, test_data = fiha.separation(instances)
print("Number of Training Instances: " + str(len(train_data)))
print("Number of Test Instances: " + str(len(test_data)))

class_prob, attrib_prob = naba.calculate_probabilities(classes, attributes, attribute_values, train_data, 0.1)
testdata_classes = naba.get_classes(class_prob, attrib_prob, test_data)
test_error = naba.calculate_error(testdata_classes)
print("Error rate: " + str(test_error))

# get mean error over k samples
mean_error = 0
k = 100000
for x in range(k):
    train_data, test_data = fiha.separation(instances)
    class_prob, attrib_prob = naba.calculate_probabilities(classes, attributes, attribute_values, train_data, 0.1)
    testdata_classes = naba.get_classes(class_prob, attrib_prob, test_data)
    mean_error += naba.calculate_error(testdata_classes)
mean_error = mean_error / k
print("Mean Error over " + str(k) + " samples: " + str(mean_error))

confusion_matrix = naba.get_confusion_matrix(classes, testdata_classes)

print("\nConfusion Matrix:")
for x in confusion_matrix:
    print(x)


