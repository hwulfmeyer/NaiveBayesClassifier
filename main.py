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

class_prob, attrib_prob = naba.calculate_probabilities(classes, attributes, attribute_values, train_data)
iprobs = naba.class_probability(class_prob, attrib_prob, test_data[0])
print(naba.choosing_of_class(iprobs, class_prob))

print("")
print(test_data[0])
print(iprobs)



