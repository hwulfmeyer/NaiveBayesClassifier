"""
This file is for executing everything together
"""
import filehandling as fiha
import naivebayes as naba


filepathnames = "datasets/cardaten/carnames.data"
filepathdata = "datasets/cardaten/car.data"

filepathnames = "datasets/tennisdaten/tennisnames.data"
filepathdata = "datasets/tennisdaten/tennis.data"

# read data
classes, attributes, attribute_values = fiha.read_data_names(filepath=filepathnames)
instances = fiha.read_data(filepath=filepathdata)
print("Number of Classes: " + str(len(classes)))
print("Number of Attributes: " + str(len(attributes)))
print("Number of Instances: " + str(len(instances)))

class_prob, attrib_prob = naba.calculate_probabilities(classes, attributes, attribute_values, instances)

print(class_prob)
for attribut in attrib_prob:
    print(attribut)


