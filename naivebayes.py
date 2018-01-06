"""
This file is for the methods concerning everything naive bayes

1. separate it in 2 clusters(1 Training , 2 Test) [DONE]
2. calculate all probabilities from training data [DONE]
3. afterwards make a function to use this probabilities and to decide to which class it is belonged [DONE]
4. calculate error rate
5. Determine the mean error rate over 100 different random samples of training data.
"""


def calculate_probabilities(classes: list, attributes: list, attribute_values: list, instances: list, smoothing: bool):
    """
    function for calculation of probabilities of classes and attributes and their corresponding classes

    :param classes: is a one-dimensional list containing the class names
    :param attributes: is one dimensional list that contains the names of attributes
    :param attribute_values: is a two-dimensional list where each row respresents
        one attribute(in the order of 'attributes') and the possible values
    :param instances: is a two-dimensional list where each row respresents
        one attribute(in the order of 'attributes') and the possible values
    :param smoothing: is a boolean parameter that decides if smoothing should be used or not
    :return:
    numclassinstances: a 2-dimensional list containg the classes, the total frequency, the number of total instances
        and the fraction of instances being that class
    attributeline: a x-dimensional list containing the probability of each attribute value to occur under a specific class
    """
    if len(instances) == 0:
        return 0, []
    classinstances = [row[-1] for row in instances]
    numclassinstances = []
    num_instances = len(classinstances)
    for dclass in classes:
        classfrequency = sum(inst[0] == dclass[0] for inst in classinstances)
        classprobline = [dclass, classfrequency, num_instances, classfrequency/num_instances]
        numclassinstances.append(classprobline)

    attributeline = []
    for i in range(0, len(attributes)):
        valueslines = []
        for k in range(0, len(attribute_values[i])):
            classprob = []
            for p in range(0, len(classes)):
                classattrbfrequency = sum((inst[-1] == classes[p] and inst[i] == attribute_values[i][k]) for inst in instances)
                if smoothing:
                    # attribute probability with laplace smoothing and assuming uniform distribution
                    attrbprobability = (classattrbfrequency + 1/len(attribute_values[i])) / (numclassinstances[p][1] + 1)
                else:
                    # without laplace smoothing
                    attrbprobability = classattrbfrequency / numclassinstances[p][1]
                classprob.append([classes[p], attrbprobability])

            valueline = [attribute_values[i][k], classprob]
            valueslines.append(valueline)

        attributeline.append([attributes[i], valueslines])

    return numclassinstances, attributeline


def get_classes(classprobs: list, attributeline: list, data: list):
    """
    function to get the predicted classes of each instance in data

    :param classprobs: a 2-dimensional list containg the classes, the total frequency, the number of total instances
        and the fraction of instances being that class
    :param attributeline: a x-dimensional list containing the probability of each attribute value to occur under a
        specific class
    :param data: is a two-dimensional list where each row respresents
        one attribute(in the order of 'attributes') and the possible values
    :return: a 1-dimensional list containg the predicted and orignal class of each instance in data
    """
    dataclasses = []
    for d in data:
        dataclasses.append([d[-1], get_class(classprobs, attributeline, d)])
    return dataclasses


def get_class(classprobs: list, attributeline: list, inputvector: list):
    """
    function to calculate the probabilties of all class of a specific instance and after that get the classes
    with the highest probability

    :param classprobs: a 2-dimensional list containg the classes, the total frequency, the number of total instances
        and the fraction of instances being that class
    :param attributeline: a x-dimensional list containing the probability of each attribute value to occur under a
        specific class
    :param inputvector: a 1-dimensional list being one instance of a specific data
    :return: the name of the class with the highest probability
    """
    probs = []
    for i in range(len(classprobs)):  # iterate over classes
        probofclass = classprobs[i][3]  # get class probability
        for j in range(len(attributeline)):  # iterate over attributes
            for k in range(len(attributeline[j][1])):  # iterate over attribute values
                if attributeline[j][1][k][0] == inputvector[j]:
                    probofclass *= attributeline[j][1][k][1][i][1]  # get value-class probability
        probs.append(probofclass)

    maxprob = 0
    index = 0
    for i in range(len(probs)):
        if probs[i] > maxprob:
            maxprob = probs[i]
            index = i
    return classprobs[index][0]


def calculate_error(dataclasses):
    return 0