"""
This file is for the methods concerning everything from file reading to file writing
"""
import re
import random


def read_data_names(filepath: str):
    """
    function to read class names & attributes

    :param filepath: the relative path to the file containing the specifications of the attribute_values
    :return: a tuple of lists
        classes: is a one-dimensional list containing the class names
        attributes: is a one-dimensional list containing the attribute names
        attribute_values: is a two-dimensional list where each row respresents
            one attribute(in the order of 'attributes') and the possible values
    """
    with open(filepath, "r") as f:
        lines = f.read().splitlines()

    classes = re.sub(r'^' + re.escape("class values: "), '', lines.pop(0))
    classes = classes.split(", ")
    attributes = re.sub(r'^' + re.escape("attributes: "), '', lines.pop(0))
    attributes = attributes.split(", ")

    attribute_values = []
    for i in range(0, len(attributes)):
        values = re.sub(r'^' + re.escape(attributes[i] + ": "), '', lines.pop(0))
        attribute_values.append(values.split(", "))

    return classes, attributes, attribute_values


def read_data(filepath: str):
    """
    function to read the actual data

    :param filepath: the relative path to the file containing the specifications of the data
    :return: the data in filepath as a two-dimensional list where each row represents one instance and
        its values
    """

    with open(filepath, "r") as f:
        lines = f.read().splitlines()
    data = []
    for line in lines:
        data.append(line.split(","))
    return data


def separation(instances):
    size_train = int((len(instances)) * 2 / 3)
    train_dataset = []
    test_set = list(instances)  # copy the full list
    while len(train_dataset) < size_train:
        index = random.randrange(len(test_set))  # find the random index to append in train data set
        train_dataset.append(test_set.pop(index))  # reduce the size of test set and increase and add the inctances in trainset
    return train_dataset, test_set
