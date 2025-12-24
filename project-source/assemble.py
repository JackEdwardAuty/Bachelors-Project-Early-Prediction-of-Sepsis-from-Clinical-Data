import numpy as np
import os
import matplotlib.pyplot as plt
import random

localPath = os.path.dirname(os.path.realpath(__file__))
recordsPath = localPath + "/records"

measurements = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
                "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST"
                "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
                "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
                "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
                "PTT", "WBC", "Fibrinogen", "Platelets", "Age", "Gender",
                "Unit1", "Unit2", "HospAdmTime", "ICULOS", "SepsisLabel"]

def writeRecord( path, UID, header=False ):
    with open(path, "r+") as currentFile:
        # print(currentFile)
        currentRecord = ""
        currentLine = 0

        for line in currentFile:
            currentLine = currentLine + 1
            if currentLine == 1:
                #need to make this happen only once
                if header == True:
                    line = line.rstrip('\n') + "|UID\n"
                    print(line)
                else:
                    continue
            else:
                line = line.rstrip('\n') + "|" + str(UID) + "\n"
            currentRecord = currentRecord + line

        # print(currentRecord)
        return currentRecord

def writeRecords( path, UIDs, UIDonly=False ): #writes specific records from collatedRecords file
    currentFile = open(path, "w")
    collatedRecords = open("collatedRecords.psv", "r")
    writtenUIDs = []

    for line in collatedRecords:
        lineList = line.rstrip('\n').split('|')
        # print(lineList[41])
        if lineList[41] in UIDs:
            # print(lineList[41])
            if UIDonly == False:
                currentFile.write(line)
            else:
                if lineList[41] in writtenUIDs:
                    continue
                else:
                    writtenUIDs.append(lineList[41])
                    currentFile.write(lineList[41] + "|")
        elif lineList[40] == "SepsisLabel":
            # print(lineList[41])
            if UIDonly == False:
                currentFile.write(line)
            else:
                # currentFile.write(lineList[41] + "|")
                continue
    currentFile.close()
    collatedRecords.close()
    print("Finished writing record " + str(path) + ", Written records: ")
    print(writtenUIDs)

def collateRecords( filename="collatedRecords.psv" ):
    recordCount = 0
    # print(localPath)
    # print(os.listdir(localPath))

    # print(recordsPath)
    recordsDirList = os.listdir(recordsPath)
    # print(recordsDirList)

    collatedRecords = open(filename , "w") #create/open collated records file

    for dir in recordsDirList:
        if os.path.isdir(recordsPath + "/" + dir):
            # print("dir " + dir)
            recordsList = os.listdir(recordsPath + "/" + dir)
            for record in recordsList:
                recordCount = recordCount + 1
                if os.path.isfile(recordsPath + "/" + dir + "/" + record):
                    recordId = int(record.split('.')[0][1:]) #takes id from filename
                    recordPath = recordsPath + "/" + dir + "/" + record
                    if recordCount == 1:
                        print(recordId)
                        collatedRecords.write(writeRecord(recordPath, recordId, header=True))
                    else:
                        collatedRecords.write(writeRecord(recordPath, recordId))

                #     print()
        else: #record
            recordCount = recordCount + 1
            recordId = int(dir.split('.')[0][1:])
            # print("not dir " + dir)
            recordPath = recordsPath + "/" + dir
            collatedRecords.write(writeRecord(recordPath, recordId))

    collatedRecords.close()#close collated records file
    print(recordCount)


def collateSepsis( records = "collatedRecords.psv" ): #isolate all records where patient diagnosed with sepsis
    collatedRecords = open( records, "r")

    sepsisUID = []
    noLines = 0
    for line in collatedRecords:
        noLines = noLines+1
        lineList = line.rstrip('\n').split('|')
        if lineList[41] in sepsisUID: #to avoid printing UID in list twice
            continue
        row = 0
        print("Current Line: " + str(noLines) + ", no. collated SepsisUIDs: " + str(len(sepsisUID)))
        if lineList[40] == '0':
                    continue
        elif lineList[40] == 'SepsisLabel':
                    continue
        else:
                    # print(value)
                    sepsisUID.append(lineList[41]) #append UID of patient to sepsis list
                    # sepsisRecords.write()
                    continue

    collatedRecords.close()

    # print(sepsisUID)
    return sepsisUID

def collateNonSepsis( records = "collatedRecords.psv", sepsisUIDs = "generate" ):
    if sepsisUIDs=="generate":
        sepsisUIDs = collateSepsis()
    else:
        sepsisUIDs = sepsisUIDs
    nonSepsisUIDs = []
    collatedRecords = open( records, "r")

    noLines = 0
    for line in collatedRecords:
        noLines = noLines+1
        print("Current Line: " + str(noLines) + ", no. collated nonSepsisUIDs: " + str(len(nonSepsisUIDs)))
        lineList = line.rstrip('\n').split('|')
        if lineList[41] in sepsisUIDs:
            continue
        elif lineList[41] in nonSepsisUIDs:
            continue
        elif lineList[41] == 'UID':
            continue
        else: #non sepsis patients
            nonSepsisUIDs.append(lineList[41])

    collatedRecords.close()
    # print(nonSepsisUIDs)
    # nonSepsisUIDs.append('1')
    return nonSepsisUIDs, sepsisUIDs

def collateHalf( seed=24444, percentSepsis=100, passThrough=False ):
    #first records a percentage of sepsisRecords to file and counts number
    #then records that number of nonSepsisRecords to file
    halfUIDs = []
    noSepsis = 0

    np.random.seed(seed) #seed the generator for reproducability

    print("Collecting a random " + str(percentSepsis) + "percent of sepsis UIDs, and the same number of nonSepsisUIDs")
    nonSepsisUIDs, sepsisUIDs = collateNonSepsis()
    print(len(sepsisUIDs))
    noSepsisUIDs = int((len(sepsisUIDs)/100)*percentSepsis)
    print(noSepsisUIDs)
    for UID in range(noSepsisUIDs):
        halfUIDs.append(random.choice(sepsisUIDs))
        print("sepsisUIDs " + str(UID/noSepsisUIDs*100) + "%, complete")


    # print(nonSepsisUIDs)
    for record in range(noSepsisUIDs):
        halfUIDs.append(random.choice(nonSepsisUIDs))
        print("nonSepsisUIDs " + str(record/noSepsisUIDs*100) + "%, complete")

    halfUIDs = set(halfUIDs) #removal of potential duplicates

    print(halfUIDs)
    if passThrough == False:
        return halfUIDs
    else:
        return sepsisUIDs, nonSepsisUIDs, halfUIDs

def collateList( data, vital, NaN ): #takes psv file and creates list of
    #a specific vital (variable) (1-42) over each line of the psv file
    valueList = []
    NaNCount = 0
    valueCount = 0

    with open(localPath + "/" + data, "r") as currentFile:
        for line in currentFile:
            lineList = line.rstrip('\n').split('|')
            col = 0
            for value in lineList:
                col = col + 1
                if col == vital:
                    valueCount = valueCount + 1
                    if value == 'NaN':
                        NaNCount = NaNCount + 1
                        if NaN == True:
                            valueList.append(value)
                        continue
                    else:
                        valueList.append(value)


    title = valueList[0]
    valueList.remove(valueList[0])

    floatValueList = []
    for value in valueList:
        if value != 'NaN':
            floatValueList.append(float(value))
        else:
            floatValueList.append(value)


    return valueList, floatValueList, NaNCount, valueCount, title


def autoBins( vital, NaN, noBins, form="bins", dataset="collatedRecords.psv" ): #makes bins
    valueList, plotList, NaNCount, valueCount, title = collateList( dataset, vital, NaN )


    counts, bins, patches = plt.hist(plotList, bins=noBins, histtype='bar', align='mid', stacked=True, density=True)
    plt.clf()
    # print(bins)
    if form == "bins":
        return bins
    elif form == "list":
        listBins = []
        for value in bins:
            listBins.append(float(round(value, 2)))
        return listBins


def makeHistogram( data, vital ):    #takes psv file and variable as input and outputs histograms to console and as files
    #vital is some number between 1 and 42, 1 HR, 2
    #make list of values from input psv file

    collatedBins = autoBins( vital, False, 15 )

    vitalList, plotList, NaNCount, valueCount, title = collateList( data, vital )
    # print(vitalList)
    # print(NaNCount)

    # print(plotList)

    plt.clf() #clear figure
    plt.figure(dpi=200, figsize=(6, 6))
    counts, bins, patches = plt.hist(plotList, bins=collatedBins, histtype='bar', align='mid', stacked=True, density=True)
    # print(bins)
    plt.title(data.rstrip('.psv') + '\n' + title + ' histogram')
    plt.figtext(0.05, 0.95, 'No Values: ' + str(valueCount) + '   No NaN: ' + str(NaNCount))
    plt.xticks(bins, rotation=90)
    plt.tick_params(axis='x', labelsize=10)

    plt.savefig(str(vital) + '_' + data.rstrip('.psv') + "_" + title + '_histogram.png')


# makeHistogram("collatedRecords.psv", 4)

def makeAllHistograms():
    localFiles = os.listdir(localPath)
    # recordsList = []
    # print(localFiles)
    for file in localFiles:
        if file.endswith('.psv'):
            # print(file)
            # recordsList.append(file)
            for i in range(1, 42):
                if i in range(35, 40):
                    continue
                print(i)
                makeHistogram(file, i)
                break #just write hr histo for eac
            # break

def makeDualHistogram( vital ): #makes a histogram with sepsis and non-sepsis data
    #on the same axes, for the specified 'vital' (1-hr, etc.)
    sepsisValueList, sepsisPlotList, sepsisNaNCount, sepsisValueCount, sepsisTitle = collateList( "sepsisRecords.psv", vital )
    nonSepsisValueList, nonSepsisPlotList, nonSepsisNaNCount, nonSepsisvalueCount, nonSepsistitle = collateList( "nonSepsisRecords.psv", vital )
    collatedBins = autoBins( vital, False, 15 )

    counts, bins, patches = plt.hist([sepsisPlotList, nonSepsisPlotList], collatedBins, label=['sepsis', 'non-sepsis'], density=True)
    plt.title('Sepsis vs non-sepsis patients ' + sepsisTitle + ' histogram')
    # plt.figtext(0.05, 0.95, 'No Values: ' + str(valueCount) + '   No NaN: ' + str(NaNCount))
    plt.legend(loc='upper right')
    plt.xticks(bins, rotation=90)
    plt.tick_params(axis='x', labelsize=10)

    plt.savefig(str(vital) + "_" + sepsisTitle + '_dual_histogram.png')

def makeAllDualHistograms(): #creates a dual density histogram for each vital
    for i in range(1, 42):
        if i in range(35, 40):
            continue
        print(i)
        makeDualHistogram( i )

def pairwiseValueRemoval( list1, list2, value ):
    #removes all cases of value in list1 along with the values of the same index in list2 and vice versa (aka deletes row)
    #could add counters, count how many NaN pairs there were compared with NaN/value pairs (how many good values are discarded)
    if len(list1) == len(list2):
        print("Starting pairwise removal of " + value + "\n")
        noValues = len(list1)
        for position in range(len(list1)-1, 0, -1):
            print("epoch 1 " + str(position) + "/" + str(noValues) + ", " + str('%.2g' % (((noValues-position)/noValues)*100)) + "% complete\n")
            if list1[position] == value:
                list1.remove(list1[position])
                list2.remove(list2[position])
        for position in range(len(list2)-1, 0, -1):
            print("epoch 2 " + str(position) + "/" + str(noValues) + ", " + str('%.2g' % (((noValues-position)/noValues)*100)) + "% complete\n")
            if list2[position] == value:
                list2.remove(list2[position])
                list1.remove(list1[position])
        return list1, list2

def strTupleConvert( strTupleList ):
    xyTupleList = []
    for coord in strTupleList:
        xyList = coord.split(',')
#         print(coord)
        xyNumTuple = ()
        for num in xyList:
#             print(float(num))
            xyNumTuple = xyNumTuple + (float(num),)
        xyTupleList.append(xyNumTuple)
    return xyTupleList

def makeScatterPlot( vitalx, vitaly, xySepsis, xyNonSepsis, dataOnly, dataset="collatedRecords.psv" ): #plots vitalx against vitaly (1-42)
    #sepsis and non-sepsis patients given a different colour
    #helps visualise boundaries and see where the majority of the data is
    datasetName = "scatterdata/" + dataset.split('.')[0]

    if dataset == "collatedRecords.psv":
        sepsisRecords = "sepsisRecords.psv"
        nonSepsisRecords = "nonSepsisRecords.psv"
        print("Using collated values from 'collatedRecords'")
    else:
        print("Using collated values files from " + datasetName)
        sepsisRecords = datasetName + "_sepsisRecords.psv"
        nonSepsisRecords = datasetName + "_nonSepsisRecords.psv"
        if xySepsis == True:
            writeRecords(datasetName + "_sepsisRecords.psv", collateSepsis( dataset )) #dataset sepsis records
        if xyNonSepsis == True:
            writeRecords(datasetName + "_nonSepsisRecords.psv", collateNonSepsis( dataset )) #dataset nonsepsis records

    print("Making scatterplot of " + measurements[vitalx-1] + str(vitalx-1) + " against " + measurements[vitaly-1] + str(vitaly-1))
    xSPlotList, ySPlotList, xNSPlotList, yNSPlotList = [], [], [], []
    if xySepsis == True:#generates scatterplotable data and stores to file

            vitalxSValueList, vitalxSPlotList, vitalxSNaNCount, vitalxSValueCount, vitalxSTitle = collateList( sepsisRecords, vitalx, True )
            # print(len(vitalxSPlotList))
            vitalySValueList, vitalySPlotList, vitalySNaNCount, vitalySValueCount, vitalySTitle = collateList( sepsisRecords, vitaly, True )
            # print(len(vitalySPlotList))

            xSPlotList, ySPlotList = pairwiseValueRemoval( vitalxSPlotList, vitalySPlotList, 'NaN' )
            print("Sepsis values cleaned")

            xySepsisList = open(datasetName + "_" + measurements[vitalx-1] + "_" + measurements[vitaly-1] + "_SepsisList.psv", "w")
            for i in range(0, len(xSPlotList)-1):
                line = str(xSPlotList[i]) + "," + str(ySPlotList[i]) + "|"
                xySepsisList.write(line)
            xySepsisList.close()

    if xyNonSepsis == True:
            vitalyNSValueList, vitalyNSPlotList, vitalyNSNaNCount, vitalyNSValueCount, vitalyNSTitle = collateList( nonSepsisRecords, vitaly, True )
            # print(len(vitalyNSPlotList))
            vitalxNSValueList, vitalxNSPlotList, vitalxNSNaNCount, vitalxNSValueCount, vitalxNSTitle = collateList( nonSepsisRecords, vitalx, True )
            # print(len(vitalxNSPlotList))

            # xyNonSepsisList = open("xySepsisList.psv", "w")
            xNSPlotList, yNSPlotList = pairwiseValueRemoval( vitalxNSPlotList, vitalyNSPlotList, 'NaN' )
            # xyNonSepsisList.close()
            print(xNSPlotList, yNSPlotList)
            print("Non-Sepsis values cleaned")

            xyNonSepsisList = open(datasetName + "_" + measurements[vitalx-1] + "_" + measurements[vitaly-1] + "_NonSepsisList.psv", "w")
            for i in range(0, len(xNSPlotList)-1):
                line = str(xNSPlotList[i]) + "," + str(yNSPlotList[i]) + "|"
                print(line)
                xyNonSepsisList.write(line)
            xyNonSepsisList.close()

    xySPlotList = []
    #take xyS from file
    with open(datasetName + "_" + measurements[vitalx-1] + "_" + measurements[vitaly-1] + "_SepsisList.psv", "r") as xySepsisList:
                for line in xySepsisList:
                    xySPlotList = line.split('|')

    if xySPlotList != []:
        print(xySPlotList.pop(len(xySPlotList)-1))#removing null end
#             print(xySPlotList)

    print("xySepsis from file, first tuple: " + xySPlotList[0])

    xyNSPlotList = []
    #take xyNS from file
    with open(datasetName + "_" + measurements[vitalx-1] + "_" + measurements[vitaly-1] + "_NonSepsisList.psv", "r") as xySepsisList:
                for line in xySepsisList:
                    xyNSPlotList = line.split('|')

    if xyNSPlotList != []:
        xyNSPlotList.pop(len(xyNSPlotList)-1)#removing null end

    print("xyNonSepsis from file, first tuple: " + xyNSPlotList[0])

    #convert to tuples
    xySTupleList = strTupleConvert(xySPlotList)
#     print(xySTupleList[0])
    xyNSTupleList = strTupleConvert(xyNSPlotList)
#     print(xyNSTupleList[0])

    if dataOnly == True:
        return xySTupleList, xyNSTupleList

    plt.clf()
    plt.figure(dpi=200)
#     xLocs, xLabels = plt.xticks(np.arange(500, step=14), xBins, rotation=90)
#     plt.tick_params(axis='x', labelsize=8)
#     yLocs, yLabels = plt.yticks(np.arange(650, step=35), yBins)
#     plt.tick_params(axis='y', labelsize=8)

    if xySTupleList:
        xyScatter = plt.scatter(*zip(*xySTupleList), s=15, alpha=0.5, c='darkorange')
#         print("50")
#         plt.title("Sepsis patients " + measurements[vitalx-1] + " against " + measurements[vitaly-1])
#         plt.savefig("scatterplots/" + datasetName + "_" + measurements[vitalx-1] + "_" + measurements[vitaly-1] + "_sepsis_scatterplot")

    if xyNSTupleList:
        xyNScatter = plt.scatter(*zip(*xyNSTupleList), s=15, alpha=0.5, c='royalblue')
        print("100")
        print(datasetName.split('/')[1])
        plt.title(datasetName.split('/')[1] + " patients " + measurements[vitalx-1] + " against " + measurements[vitaly-1])
        plt.legend([xyScatter, xyNScatter], ['sepsis', 'non-sepsis'], loc='upper right')
        plt.savefig("scatterplots/" + datasetName + "_" + measurements[vitalx-1] + "_" + measurements[vitaly-1] + "_scatterplot.png")

    return xySTupleList, xyNSTupleList

def makeScatterPlotWithHistograms( vitalx, vitaly, dataset="halfRecords.psv", generate=False ):
    datasetName = dataset.split('.')[0]

    #definitions for axes
    left, width = 0.1, 0.6
    bottom, height = 0.1, 0.6
    spacing = 0.01

    scatter_dimensions = [left, bottom, width, height]
    histX_dimensions = [left, bottom+height+spacing, width, 0.175]
    histY_dimensions = [left+width+spacing, bottom, 0.175, height]

    plt.figure(figsize=(10,10))

    if generate==False:
        xySTupleList, xyNSTupleList = makeScatterPlot( vitalx, vitaly, False, False, dataOnly=True, dataset=dataset )
    else:
        xySTupleList, xyNSTupleList = makeScatterPlot( vitalx, vitaly, True, True, dataOnly=True, dataset=dataset )
    scatterAxes = plt.axes(scatter_dimensions)

#     xLocs, xLabels = plt.xticks(xBins, rotation=90)
    plt.tick_params(axis='x', labelsize=8)

#     yLocs, yLabels = plt.yticks(yBins)
    plt.tick_params(axis='y', labelsize=8)

    xyScatter = scatterAxes.scatter(*zip(*xySTupleList), s=15, alpha=0.5, c='darkorange')
    xyNScatter = scatterAxes.scatter(*zip(*xyNSTupleList), s=15, alpha=0.5, c='royalblue')

    #calculating 'bins' for axes
    xNoBins = 10
#     xBins = autoBins( vitalx, False, xNoBins, form="list", dataset=dataset ) #autobins clears figure
#     print(xBins)
    yNoBins = 10
#     yBins = autoBins( vitaly, False, yNoBins, form="list", dataset=dataset )
#     print(yBins)
    #calculating bins using scatterplot limits
    xLims = scatterAxes.get_xlim()
    yLims = scatterAxes.get_ylim()
    xBins = np.arange(start=int(xLims[0]), stop=int(xLims[1]), step=(xLims[1]-xLims[0])/xNoBins)
    print(xBins)
    print(yLims)
    print(yLims[1]-yLims[0])
    print(yNoBins)
    print((yLims[1]-yLims[0])/yNoBins)
    yBins = np.arange(start=int(yLims[0]), stop=int(yLims[1]), step=(yLims[1]-yLims[0])/yNoBins)
    print(yBins)

    #Histograms
    histXAxes = plt.axes(histX_dimensions)
    histXAxes.set_xlim(scatterAxes.get_xlim())

    histXAxes.tick_params(direction='in', labelbottom=False)
    xSepsisValueList, xSepsisPlotList, xSepsisNaNCount, xSepsisValueCount, xSepsisTitle = collateList( "sepsisRecords.psv", vitalx, NaN=False )
    xNonSepsisValueList, xNonSepsisPlotList, xNonSepsisNaNCount, xNonSepsisvalueCount, xNonSepsistitle = collateList( "nonSepsisRecords.psv", vitalx, NaN=False )
    histXAxes.hist([xSepsisPlotList, xNonSepsisPlotList], xBins, color=['darkorange', 'royalblue'], label=['sepsis', 'non-sepsis'], density=True)

    plt.title(measurements[vitalx-1] + " against " + measurements[vitaly-1] + " Scatterplot with Histograms", loc='right')


    histYAxes = plt.axes(histY_dimensions)
    histYAxes.set_ylim(scatterAxes.get_ylim())

    histYAxes.tick_params(direction='in', labelleft=False)
    ySepsisValueList, ySepsisPlotList, ySepsisNaNCount, ySepsisValueCount, ySepsisTitle = collateList( "sepsisRecords.psv", vitaly, NaN=False )
    yNonSepsisValueList, yNonSepsisPlotList, yNonSepsisNaNCount, yNonSepsisvalueCount, yNonSepsistitle = collateList( "nonSepsisRecords.psv", vitaly, NaN=False )
    histYAxes.hist([ySepsisPlotList, yNonSepsisPlotList], yBins, color=['darkorange', 'royalblue'], label=['sepsis', 'non-sepsis'], density=True, orientation='horizontal')

    scatterAxes.legend([xyScatter, xyNScatter], ['sepsis', 'non-sepsis'], loc='upper right')
    plt.savefig("scatterplots/withHistograms/" + datasetName + "_" + measurements[vitalx-1] + "_" + measurements[vitaly-1] + "_scatterwithhistograms.png")

def makeNNSets( seed=3, filename = "collatedRecords.psv", testSet=False ):
    trainingSet, validationSet, testSet = {}, {}, {}
    #create training set
    sepsisUIDs, nonSepsisUIDs, halfUIDs = collateHalf( seed=seed, percentSepsis=70, passThrough=True )
    trainingSet = halfUIDs
    print("Collated training set")
    remainingSepsisUIDs = sepsisUIDs
    for UID in halfUIDs:
        if UID in sepsisUIDs:
            sepsisUIDs.remove(UID)
        if UID in nonSepsisUIDs:
            nonSepsisUIDs.remove(UID)

    if testSet == False:
        #create test set from remaining data
        validationSet = set(nonSepsisUIDs).union(set(sepsisUIDs))
        print("Collated validation set")
        return trainingSet, validationSet
    else:
        #create balanced validation and test sets
        validationList = []
        np.random.seed(seed)
        noRemainingSepsisUIDs = len(sepsisUIDs)
        noRemainingNonSepsisUIDs = len(nonSepsisUIDs)
        for UID in range(int(noRemainingSepsisUIDs/2)):
            valUID = random.choice(sepsisUIDs)
            validationList.append(valUID)
            sepsisUIDs.remove(valUID)
        for UID in range(int(noRemainingNonSepsisUIDs/2)):
            valUID = random.choice(nonSepsisUIDs)
            validationList.append(valUID)
            nonSepsisUIDs.remove(valUID)
        validationSet = set(validationList)
        print("Collated validation set")

        testSet = set(nonSepsisUIDs).union(set(sepsisUIDs))
        print("Collated test set")
        return trainingSet, validationSet, testSet


# collateRecords()
# sepsisUIDs, nonSepsisUIDs, halfUIDs = collateHalf(seed=1234, passThrough=True)
# writeRecords("sepsisRecords.psv", sepsisUIDs)
# writeRecords("nonSepsisRecords.psv", nonSepsisUIDs)
# writeRecords("halfRecords.psv", halfUIDs)
# makeAllHistograms()
# makeAllDualHistograms()
# makeScatterPlotWithHistograms( 1, 4, generate=False )
# makeScatterPlotWithHistograms( 1, 3, generate=True )

# makeScatterPlotWithHistograms( 1, 11, generate=False )
# makeScatterPlotWithHistograms( 1, 12, generate=False )
# makeScatterPlotWithHistograms( 1, 24, generate=False )
# makeScatterPlotWithHistograms( 1, 22, generate=False )
# xySTupleList, xyNSTupleList = makeScatterPlot( 1, 3, True, True, False )
# print(xySTupleList[len(xySTupleList)-1])
# print(xyNSTupleList[len(xyNSTupleList)-1])

makeScatterPlot( 1, 3, False, False, False, dataset="halfRecords.psv" )

# trainingSet, validationSet, testSet = makeNNSets( seed=808, testSet=True )
# print("Made NN sets, writing...")
# writeRecords( "trainingSet.psv", trainingSet, UIDonly=True )
# writeRecords( "validationSet.psv", validationSet, UIDonly=True )
# writeRecords( "testSet.psv", testSet, UIDonly=True )
