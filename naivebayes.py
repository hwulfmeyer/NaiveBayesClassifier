"""
This file is for the methods concerning everything naive bayes

1. separate it in 2 clusters(1 Training , 2 Test)
2. calculate all probabilities from training data [DONE]
3. afterwards make a function to use this probabilities and to decide to which class it is belonged
"""


def calculate_probabilities(classes: list, attributes: list, attribute_values: list, instances: list):
    """

    :param classes: is a one-dimensional list containing the class names
    :param attributes: is one dimensional list that contains the names of attributes
    :param attribute_values: is a two-dimensional list where each row respresents
            one attribute(in the order of 'attributes') and the possible values
    :param instances: is a two-dimensional list where each row respresents
          one attribute(in the order of 'attributes') and the possible values
    :return:
    numclassinstances: a 2-dimensional list containg the classes, the total frequency, the number of total instances and the
    fraction of instances being that class
    attributeline:  a 4-dimensional list containg the attributes with their values and how many of these values correespond
    to a specific class
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
                classprob.append([classes[p], classattrbfrequency/numclassinstances[p][1] ])

            valueline = [attribute_values[i][k], classprob]
            valueslines.append(valueline)

        attributeline.append([attributes[i], valueslines])

    return numclassinstances, attributeline


