from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pylab import *
import pandas as pd
import pprint
import itertools
import random

COUNTRIES = {
    "Balinese": "Bali",
    "Bolivian": "Bolivia",
    "Brazilian": "Brazil",
    "Costa Rican": "Costa Rican",
    "Dominican": "Dominican Republic",
    "Salvadorean": "El Salvador",
    "Ethiopian": "Ethiopia",
    "Guatemalan": "Guatemala",
    "Indian": "India",
    "Kenyan": "Kenya",
    "Malian": "Mali",
    "Mexican": "Mexico",
    "Panamanian": "Panama",
    "Peruvian": "Peru",
    "Sumatran": "Sumatra",
}

PROPERTIES = "Decaf,Organic,Fair Trade"

class Coffee(object):
    @classmethod
    def fromname(cls, name):
        parse(name)

def parse(name):
    '''
    parses a given coffee name into its individual properties and values.
    '''
    print(name)

    #parsing properties and adjective
    propertiesToCheck = PROPERTIES.split(",")
    propertyDict = {"Decaf":"False","Organic":"False","Fair Trade":"False","Adjective":"","Country":""}

    # multiWordProperty handles case like Fair Trade
    multiWordProperty = ""
    currPos = 0
    nameVals = name.split(" ")
    for w in nameVals:
        if w in PROPERTIES:
            if(w in propertiesToCheck):
                propertyDict[w] = "True"
            else:
                multiWordProperty = w if multiWordProperty == "" else multiWordProperty + " " + w
            currPos = currPos + 1

    currCountryIndex = 1
    country = nameVals[-1]

    if(multiWordProperty != ""):
        propertyDict[multiWordProperty] = "True"

    while((country in COUNTRIES) == False):
        currCountryIndex = currCountryIndex + 1
        country = nameVals[-currCountryIndex] + " " + country

    propertyDict["Country"] = COUNTRIES[country]

    propertyDict["Adjective"] = nameVals[currPos]
    while(currPos < len(nameVals)-currCountryIndex-1):
        currPos = currPos + 1
        propertyDict["Adjective"] = propertyDict["Adjective"] + " " + nameVals[currPos]

    for k,v in propertyDict.items():
        tab = "\t\t" if len(k) < 6 else "\t"
        print("\t",k,tab,v)
    return propertyDict


def parse_no_print(name):

    #parsing properties and adjective
    propertiesToCheck = PROPERTIES.split(",")

    propertyDict = {"Decaf":"False","Organic":"False","Fair Trade":"False","Adjective":"","Country":""}

    # multiWordProperty handles case like Fair Trade
    multiWordProperty = ""
    currPos = 0
    nameVals = name.split(" ")
    for w in nameVals:
        if w in PROPERTIES:
            if(w in propertiesToCheck):
                propertyDict[w] = "True"
            else:
                multiWordProperty = w if multiWordProperty == "" else multiWordProperty + " " + w
            currPos = currPos + 1

    currCountryIndex = 1
    country = nameVals[-1]

    if(multiWordProperty != ""):
        propertyDict[multiWordProperty] = "True"

    while((country in COUNTRIES) == False):
        currCountryIndex = currCountryIndex + 1
        country = nameVals[-currCountryIndex] + " " + country

    propertyDict["Country"] = COUNTRIES[country]

    propertyDict["Adjective"] = nameVals[currPos]
    while(currPos < len(nameVals)-currCountryIndex-1):
        currPos = currPos + 1
        propertyDict["Adjective"] = propertyDict["Adjective"] + " " + nameVals[currPos]

    return propertyDict



def summarize(file):
    '''
    collects and prints out the counts of various properties along with the total number
    of users and coffee types in the given data set.
    '''
    # stats holds the summary information to be printed
    stats = {'ids':[],'coffeeTypes':[],'Country':{},'Adjective':{},'Decaf':{'True':0,'False':0},'Organic':{'True':0,'False':0},'Fair Trade':{'True':0,'False':0}}

    # user_info holds the mapping from users to coffees and coffee ratings and properties
    userInfo = {}

    file_obj = open(file.name,file.mode)
    numRatings = 0
    for line in file_obj:
        words = line.split()
        # NumRatings
        numRatings = numRatings + 1
        currUserId = words[0]
        if currUserId not in stats["ids"]:
            stats["ids"].append(currUserId)
            userInfo[currUserId] = {}
            userInfo[currUserId]["coffees"] = []
            userInfo[currUserId]["ratings"] = []

        coffeeType = ' '.join(words[1:-1])
        properties = parse_no_print(coffeeType)
        currRating = words[-1]

        userInfo[currUserId]["coffees"].append(coffeeType)
        userInfo[currUserId]["ratings"].append(int(currRating))

        if coffeeType not in stats["coffeeTypes"]:
            stats["coffeeTypes"].append(coffeeType)
            userInfo[coffeeType] = []

        for k,v in properties.items():
            if k in ("Decaf","Organic","Fair Trade"):
                stats[k][v] = stats[k][v] + 1
            if k in ("Adjective","Country"):
                if v not in stats[k]:
                    stats[k][v] = 1
                else:
                    stats[k][v] = stats[k][v] + 1

    stats["TotalRatings"] = numRatings
    percentValues = print_summary(stats)
    return userInfo,stats,percentValues


def summarize_no_print(file):
    # stats holds the summary information to be printed
    stats = {'ids':[],'coffeeTypes':[],'Country':{},'Adjective':{},'Decaf':{'True':0,'False':0},'Organic':{'True':0,'False':0},'Fair Trade':{'True':0,'False':0}}

    # user_info holds the mapping from users to coffees and coffee ratings and
    # properties
    userInfo = {}

    file_obj = open(file.name,file.mode)
    numRatings = 0
    for line in file_obj:
        words = line.split()
        # NumRatings
        numRatings = numRatings + 1
        currUserId = words[0]
        if currUserId not in stats["ids"]:
            stats["ids"].append(currUserId)
            userInfo[currUserId] = {}
            userInfo[currUserId]["coffees"] = []
            userInfo[currUserId]["ratings"] = []

        coffeeType = ' '.join(words[1:-1])
        properties = parse_no_print(coffeeType)
        currRating = words[-1]

        userInfo[currUserId]["coffees"].append(coffeeType)
        userInfo[currUserId]["ratings"].append(int(currRating))

        if coffeeType not in stats["coffeeTypes"]:
            stats["coffeeTypes"].append(coffeeType)
            userInfo[coffeeType] = []

        for k,v in properties.items():
            if k in ("Decaf","Organic","Fair Trade"):
                stats[k][v] = stats[k][v] + 1
            if k in ("Adjective","Country"):
                if v not in stats[k]:
                    stats[k][v] = 1
                else:
                    stats[k][v] = stats[k][v] + 1

    stats["TotalRatings"] = numRatings
    return userInfo,stats


def print_summary(stats):
    # print the summarized results
    print("Total people: ",len(stats["ids"]))
    print("Total coffee types: ",len(stats["coffeeTypes"]))
    properties = ["Decaf","Organic","Fair Trade"]
    labels = "True","False"
    percentValues = [];
    for p in properties:
        print(p);
        print("\t",labels[0],":", stats[p][labels[0]])
        print("\t",labels[1],":", stats[p][labels[1]])
        # percentValues for visualize
        percentValues.append([stats[p]["True"]/stats["TotalRatings"],stats[p]["False"]/stats["TotalRatings"]])

    print("Adjective:")
    for k,v in stats["Adjective"].items():
        print("\t",k,v)

    print("Country:")
    for k,v in stats["Country"].items():
        print("\t",k,v)

    return percentValues

def sample(file):
    """
    The given dataset is split 75% and 25% into training set
    and test set
    """

    f = open(file.name,file.mode)
    f1 = open("training_set.txt",'w')
    f2 = open("test_set.txt",'w')
    i = 0
    for line in f:
        if i%4 == 0:
            f2.write(line)
        else:
            f1.write(line)
        i = i + 1
    f.close()
    f1.close()
    f2.close()


def create_weight_matrix(userInfo,stats):
    """
    The basic idea used for recommendation is a kind of content based recommendation per user.
    It involves creating a weight matrix (user across coffee properties). The weights are basically
    estimates of ratings by the user depending on the properties of the coffee.
    The individual weights are calculated using the average rating given by this
    user to coffees containing each of the 5 different properties.
    """
    wm = {}
    properties = ["Decaf","Organic","Fair Trade"]
    for currUserId in stats["ids"]:
        wm[currUserId]={}
        for p in properties:
            userInfo[currUserId][p] = {}
            # extracts all the current users rated coffees containing
            # decaf,organic or fair trade property
            userInfo[currUserId][p]["coffees"] = list(coffees for coffees in userInfo[currUserId]["coffees"] if p in coffees)
            userInfo[currUserId][p]["rating"] = list(itertools.compress(userInfo[currUserId]["ratings"], userInfo[currUserId][p]["coffees"]))
            #computes the average coffee rating by the user for a coffee containing the property
            wm[currUserId][p] = sum(userInfo[currUserId][p]["rating"])/len(userInfo[currUserId][p]["rating"])

        for attribute in ("Adjective","Country"):
            wm[currUserId][attribute] = {}
            for k,v in stats[attribute].items():
                curList = list(coffees for coffees in userInfo[currUserId]["coffees"] if k in coffees)
                userInfo[currUserId][k] = {}
                if(len(curList) > 0):
                    userInfo[currUserId][k]["coffees"] = curList
                    userInfo[currUserId][k]["rating"] = list(itertools.compress(userInfo[currUserId]["ratings"], userInfo[currUserId][k]["coffees"]))
                    wm[currUserId][attribute][k] = sum(userInfo[currUserId][k]["rating"])/len(userInfo[currUserId][k]["rating"])
                else:
                    wm[currUserId][attribute][k] = 0
    return wm


def predict_ratings(userid,coffee,wm):
    '''
    given a userid and an unrated coffee, use the weight matrix to
    predict the ratings of a coffee. In our case, all the properties
    are given equal importance.
    '''
    properties = parse_no_print(coffee)
    rating = 0
    count = 0
    for k,v in properties.items():
        if k in ('Decaf','Organic','Fair Trade'):
            if v == "True":
                rating = rating + wm[userid][k]
                count = count + 1
        else:
            if wm[userid][k][v] > 0:
                rating = rating + wm[userid][k][v]
                count = count + 1
    return int(math.ceil(rating/count)) if count > 0  else 0


def recommend(file):
    '''
    Sampling is done to create a training set and this data set is parsed to generate a weight matrix.
    In the weight matrix, weights are estimated for all the 5 extracted coffee properties (decaf,
    fair trade, organic, adjective and country) for a given user. The weights are estimated based
    on the 'average rating' given by the user for a coffee containing each of the above coffee
    properties from the parsed dataset.An unrated coffee list is then created for each user,
    and the unrated coffee ratings are predicted using the weight matrix. For the sake of simplicty
    all the properties are given equal importance in predicting the rating. These predicted coffee
    ratings are sorted and the top 3 ratings are then recommended to that user.
    '''
    sample(file)
    f = open("training_set.txt","r")
    userInfo,stats = summarize_no_print(f)
    wm = create_weight_matrix(userInfo,stats)

    for currUserId in stats["ids"]:
        unratedCoffees = (unratedCoffees for unratedCoffees in stats["coffeeTypes"] if unratedCoffees not in userInfo[currUserId]["coffees"])
        unratedCoffees = list(unratedCoffees)
        topRatedCoffees = []
        topRatedCoffeeRatings = []

        for coffee in unratedCoffees:
            coffee_predicted_rating = predict_ratings(currUserId,coffee,wm)

            if(coffee_predicted_rating > 0):
                topRatedCoffees.append(coffee)
                topRatedCoffeeRatings.append(coffee_predicted_rating)

        combinedRatedCoffees = sorted(zip(topRatedCoffeeRatings,topRatedCoffees),reverse=True)
        topRatedCoffeeRatings = [val[0] for val in combinedRatedCoffees]
        topRatedCoffees = [val[1] for val in combinedRatedCoffees]

        for i in range(0,3):
            print(currUserId,topRatedCoffees[i],topRatedCoffeeRatings[i])


def visualize(file):
    '''
    use matplotlib to visually display the summary information  for coffee_ratings.txt dataset
    '''

    userInfo,stats,percentValues = summarize(file)
    title("Coffee Summary")
    properties = ["Decaf","Organic","Fair Trade"]
    labels = "True","False"

    # create pi charts to show ratio of true/false occurances of decaf,organic
    # and fair trade properties
    figure(1, figsize=(30,15))
    for i in [0,1,2]:
        axes([0.33*0.8*(i+1)-0.15, 0.4, 0.8*0.33, 0.8*0.33],aspect='equal')
        pie(percentValues[i], labels=labels,colors=['gold', 'lightskyblue'],autopct='%1.f%%', shadow=True,
            startangle=0,radius=0.8,center=(0.9,0.5),labeldistance=1.4)
        title(properties[i])
        legendlabels = stats[properties[i]][labels[0]],stats[properties[i]][labels[1]]
        legend(title="# " + properties[i] + " Rated",labels=legendlabels, loc=(0.05 - i*0.05,-0.55))

    # create bar graph of Countries
    percentValues = []
    labels = []
    for k,v in stats["Country"].items():
        print("\t",k,v)
        labels.append(k)
        percentValues.append(100*(int(v)/stats["TotalRatings"]))

    figure(2,figsize=(30,15))

    s = pd.Series(percentValues,index = labels)

    title("Percent Rated Per Country")
    ylabel('Percent')
    xlabel('Country')

    s.plot(kind='bar',color='rgbkymc')

    # create bar graph of Adjectives
    percentValues = []
    labels = []
    for k,v in stats["Adjective"].items():
        print("\t",k,v)
        labels.append(k)
        percentValues.append(100*(int(v)/stats["TotalRatings"]))

    figure(3,figsize=(30,15))

    s = pd.Series(percentValues,index = labels)

    title("Percent Rated Per Adjective")
    ylabel('Percent')
    xlabel('Adjective')

    s.plot(kind='bar',color='rgbkymc')

    show()


def main():
    parser = argparse.ArgumentParser(description='TextNow Coffee Tasting')
    subparsers = parser.add_subparsers(dest='command', help='command')

    commands = ['parse', 'summarize', 'recommend', 'visualize']
    parsers = {
        c: subparsers.add_parser(c) for c in commands
    }

    parsers['parse'].add_argument('arg', help='coffee descriptive name')
    parsers['summarize'].add_argument('arg', help='input csv file',
                                      type=argparse.FileType('r'))
    parsers['recommend'].add_argument('arg', help='input csv file',
                                      type=argparse.FileType('r'))
    parsers['visualize'].add_argument('arg', help='input csv file',
                                      type=argparse.FileType('r'))

    args = parser.parse_args()
    globals()[args.command](args.arg)


if __name__ == '__main__':
    main()
