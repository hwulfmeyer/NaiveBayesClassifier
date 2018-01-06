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

class_prob, attrib_prob = naba.calculate_probabilities(classes, attributes, attribute_values, train_data, True)
testdata_classes = naba.get_classes(class_prob, attrib_prob, test_data)
print(testdata_classes)
test_error = naba.calculate_error(testdata_classes)




