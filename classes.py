import torch
import numpy as np
from torch import nn
from torch.utils import data

class SepsisModel(nn.Module):
  def __init__(self, nCols):
    #defines operations for forward method
    super().__init__()
    self.hiddenLayerSize = int(nCols/2)
    print(nCols, self.hiddenLayerSize)

    #input layer to hidden layer transformation
    self.hidden = nn.Linear(nCols, self.hiddenLayerSize)

    self.ReLU = nn.ReLU()

    #hidden to output layer transformation
    self.output = nn.Linear(self.hiddenLayerSize, 1)

    #activation function
    self.sigmoid = nn.Sigmoid()

    #output softmax function
    self.softmax = nn.Softmax(dim=1)

    #dropout function
    self.dropout2 = nn.Dropout(0.2) #20% chance to dropout

    self.dropout5 = nn.Dropout(0.5) #50% chance to dropout


  def forward(self, x): #same as building sequential model of above operations
    #takes x and passes through operations
    #need to modify
    x = self.dropout2(x)

    # print(x)
    c1 = self.hidden(x)
    # print(c1)
    c2 = self.softmax(c1)
    # print(c2)

    drop = self.dropout2(c2)

    # print(sig1)
    out = self.output(drop)
    # print(out)

    sig1 = self.sigmoid(out)
    # print(sig2)
    return sig1

class SepsisDataset(data.Dataset):
  def __init__(self, UIDList, colSequence):
    #initialises dataset
    self.UIDs = UIDList #takes list of UIDs from input parameters
    #loads data from file
    self.colSequence = colSequence

    self.unusableRecordCount = 0

  def __len__(self):
    #gives total number of samples
    return len(self.UIDs)

  def __getitem__(self, index):
    #generates one data sample
    #select sample
    UID = self.UIDs[index]

    #load data and get label
    if int(UID) > 100000:
        path = 'records/training_setB/' + 'p' + str(UID) + '.psv'
    else:
        path = 'records/training/' + 'p' + str(UID).zfill(6) + '.psv'
    # print(path)


    xArray = np.loadtxt(path, delimiter='|', dtype=np.float64,
                        skiprows=1, usecols=self.colSequence)

    yArray = np.loadtxt(path, delimiter='|', dtype=np.float64,
                        skiprows=1, usecols=(39, 40))

    noDimensions = len(self.colSequence)
    npVariable = xArray[:, 0:noDimensions]
    npLabels = yArray[:, 1:2]


    # find rows with NaNs and record ICULOS in a list for row for removal from tensor
    #can either handle NaN values by removing entire row
    #or by calculating a median value (for continuous data) between the two adjacent value
    # noNanRows = 0
    noRows = len(npVariable)

    nanRows = []
    for i in range(noRows):
        for value in npVariable[i]:
            if str(value.item()) == 'nan':
                nanRows.append(i)
                break
    # print(nanRows)

    if len(nanRows) > 0:
        # print("\n\nUID", UID, nanRows)
        npVariable, npLabels = removeRows(npVariable, npLabels, nanRows, path=path, colSequence=self.colSequence)
        # print("unusableRecordCount", unusableRecordCount)
        # print("\nUID", UID, nanRows)

    if len(npVariable) > 0:
        #normalise dataset
        npVariable = normalize(npVariable)

    variables = torch.from_numpy(npVariable).float()
    labels = torch.tensor(npLabels).float()

    x = variables
    y = labels

    return x, y, UID


def normalize(npVariable): #takes input array and normalizes so that values lie between 0 and 1.

        valueListArray = []
        for i, row in enumerate(npVariable):
            valueList = []
            for j, value in enumerate(row):
                valueList.append((j, value))
            valueListArray.append(valueList)
        # print("valueListArray:", valueListArray)

        #find min and max values for each col
        maxValueList = valueListArray[0].copy()
        # print(maxValueList)
        minValueList = valueListArray[0].copy()
        # print(minValueList)
        for valueList in valueListArray:
            # print(valueList)
            for j, value in valueList:
                # print(value, maxValueList[j][1])
                if value > maxValueList[j][1]:
                    # print("value larger:", j, value, maxValueList[j][1])
                    maxValueList[j] = (j, value)
                if str(maxValueList[j][1]) == 'nan':
                    # print("nan:", j, maxValueList[j][1])
                    if str(value) != 'nan':
                        maxValueList[j] = (j, value)

                # print(value, minValueList[j][1])
                if value < minValueList[j][1]:
                    # print("value smaller:", j, value, minValueList[j][1])
                    minValueList[j] = (j, value)
                if str(minValueList[j][1]) == 'nan':
                    # print("nan:", j, minValueList[j][1])
                    if str(value) != 'nan':
                        minValueList[j] = (j, value)
        # print("updateMaxValueList", maxValueList)
        # print("updateMinValueList", minValueList)

        #applying normalisation using (x-min(x))/(max(x)-min(x))
        # print("prenormalised array:", npVariable)
        for i, row in enumerate(npVariable):
            # print("prenormalised row:", row)
            for j, value in enumerate(row):
                # print("prenormalised value:", row[j])
                row[j] = (row[j]-minValueList[j][1])/(maxValueList[j][1]-minValueList[j][1])
                # print("normalised value:", row[j])
            # print("normalised row:", row)
        print("normalised array:", npVariable)


        return npVariable


unusableRecordCount = 0

def removeRows( array, labels, nanRows, path=None, colSequence=None ): #takes a list of ICULOS(rows) as input with a numpy array
        #deals with the rows on a case by case basis - either delete or calculate median between
        #need to further improve this need to measure
        #how many rows are being removed total, get % might be removing too many rows.
        #also find which cols are most causing removal - definitely highest nan% cols
        # print(labels)
        # print(array)
        # print(nanRows)
        remainingNanRows = nanRows.copy() #without copy just references original
        # print(remainingNanRows)
        # print(list(enumerate(nanRows)))

        removedRows, updatedRows = [], []
        for i, row in enumerate(nanRows):
            # print(i+1, "/", len(nanRows), "nanRows", nanRows)
            index = i - len(removedRows) #adjusts index
            # print(index+1, "/", len(remainingNanRows), "remainingNanRows:", remainingNanRows)

            # print("i:", i, ",row:", row, "noRemoved:", noRemoved, ",index:", index)
            # print("arrayindex", index, ":", array[index])

            #check how many of the values of the current row are nan values
            nanCols = nanList(array[index], colSequence)
            # print("cols with nan in this row:", nanCols)

            noNan = len(nanCols)
            # print("noNan: " + str(noNan))

            noCols = len(array[:][0])

            if noNan >= int(noCols/2): #if more than 50% the values are nan delete row
                # print("Deleting row(index): ", array[index])
                array = np.delete(array, index, 0)
                labels = np.delete(labels, index, 0)
                # print("Row " + str(index) + " deleted.")
                # print(noNan, "NaN cols,", int(noCols/1.5), " >1/1.5 nan, deleted row,", i)
                # print("index:", index, remainingNanRows)
                removedRows.append(remainingNanRows[i]) #should append row
                # print("post removal remainingNanRows:", remainingNanRows)
                continue

            # print(noNan)
            prevIndex, nextIndex = findClosestViableRows(array, i, index, remainingNanRows, nanList, colSequence)
            print(prevIndex,nextIndex)


            if int(prevIndex) and int(nextIndex) != -1: #still allows nan to propogate somehow
            #for some reason is allowing cases where prevIndex is -1
                # print(index, "viable prev/next index", prevIndex, nextIndex)
                # print("Calculating mean values for:",array[index],"from",array[prevIndex],"and",array[nextIndex])

                for col, value in enumerate(array[index]):
                    # print(col, value)
                    if str(value.item()) == 'nan':
                        # print("Calculating mean value for:",array[index][col],"from",array[prevIndex][col],"and",array[nextIndex][col])
                        meanValue = (float(array[prevIndex][col]) + float(array[nextIndex][col]))/2
                        # print("mean value: " + str(meanValue))
                        array[index][col] = meanValue
                # print("updated:", array[index])
                updatedRows.append(remainingNanRows[i])
                # print("post update remainingNanRows:", remainingNanRows)
                continue
            else: #if either arent viable delete row..
                # print("index", index, "remainingNanRows", remainingNanRows)
                removedRows.append(remainingNanRows[i])
                # print(index, "no viable prev/next index", prevIndex, nextIndex)
                array = np.delete(array, index, 0)
                labels = np.delete(labels, index, 0)

                # print("post removal remainingNanRows:", remainingNanRows)
                continue

        # print("removedRows:", removedRows)
        # print("updatedRows:", updatedRows)
        print(remainingNanRows)
        remainingNanRows = [x for x in remainingNanRows if x not in removedRows]
        # print("remainingNanRows after removedRows:", remainingNanRows)
        remainingNanRows = [x for x in remainingNanRows if x not in updatedRows]

        # print("remainingNanRows (post):", remainingNanRows) #should be empty

        # print(noRemoved)
        #majority of incoming arrays become len= 0 by this point - not acceptable

        if len(array) == 0:
            # print("Whole record unusable: " + path) #should do more
            # path = path
            # print("unusable")
            # self.unusableRecordCount += 1
            global unusableRecordCount
            unusableRecordCount += 1
        #     print("array", array, len(array))
        # else:
        #     print("row1:",array[:][0])


        noRows = len(array)
        print("noRows: " + str(noRows))
        print("noRemoved:", len(removedRows))

        print(array)
        # print(labels)
        return array, labels


def findClosestViableRows(array, i, index, remainingNanRows, nanCols, colSequence, prev=True, next=True):
        # print(remainingNanRows)
        # print("index:", index, "row:", array[index])
        # print("ICULOS", array[index][-1:])
        prevViable, nextViable = True, True
        tooBig, tooSmall = False, False

        indexNanSet = set(nanList( array[index], colSequence ))
        # print("indexNanSet:", indexNanSet)

        # col specific actions using nanList function
        if prev == True:
            prevIndex = index - 1

            if prevIndex < 0:
                prevViable = False
                tooSmall = True
            else:
                if prevIndex in remainingNanRows:
                    prevViable = False

                # nextIndexNanSet
                sharedNanElements, colSpecific = colSpecificActions(array[prevIndex], indexNanSet, colSequence)
                nonSharedNanElements = indexNanSet - sharedNanElements
                # print("indexNanSet", indexNanSet)
                # print("sharedNanElements", sharedNanElements)
                # print("nonSharedNanElements", nonSharedNanElements)

                if len(sharedNanElements) > 0:
                    if len(sharedNanElements) > len(colSpecific):
                        prevViable = False
                    else: #all sharedNanElements are those with colSpecific actions
                        prevViable = True
                        print(sharedNanElements, colSpecific)
                else:
                    prevViable = True #allows no shared nan element


            if prevViable == False and tooSmall == False:
                prevIndex = findClosestViableRows(array, i, index-1, remainingNanRows, nanCols, colSequence, prev=True, next=False)
                if (prevIndex != -1):
                    prevViable = True
                    # print("recurse prev:", index)


            if prevViable:
                prevIndex = prevIndex
                # print("found viable prev index", prevIndex)
            else:
                prevIndex = -1
                # print("not found viable prev index", prevIndex)


        if next == True:
            nextIndex = index + 1

            if nextIndex > len(array)-1:
                nextViable = False
                tooBig = True
            else:
                if nextIndex in remainingNanRows:
                    nextViable = False

                sharedNanElements, colSpecific = colSpecificActions(array[nextIndex], indexNanSet, colSequence)

                if len(sharedNanElements) > 0:
                    if len(sharedNanElements) > len(colSpecific):
                        nextViable = False
                    else: #all sharedNanElements are those with colSpecific actions
                        nextViable = True
                        print(sharedNanElements, colSpecific)

                else:
                    nextViable = True #allows no shared nan element



            if nextViable == False and tooBig == False:
                nextIndex = findClosestViableRows(array, i, index+1, remainingNanRows, nanCols, colSequence, prev=False, next=True)
                if (nextIndex != -1):
                    nextViable = True
                    # print("recurse next:", index)


            if nextViable:
                nextIndex = nextIndex
                # print("found viable next index", nextIndex)

            else:
                nextIndex = -1
                # print("not found viable next index", nextIndex)

        if next and prev == True:
            if nextViable:
                prevIndex = prevIndex
                # print("found viable next index", nextIndex)
            else:
                nextIndex = -1
                # print("not found viable next index", nextIndex)

            if prevViable:
                prevIndex = prevIndex
                # print("found viable prev index", prevIndex)
            else:
                prevIndex = -1
                # print("not found viable prev index", prevIndex)
            return prevIndex, nextIndex
        elif next == True:
            return nextIndex
        elif prev == True:
            return prevIndex

def nanList( row, colSequence ):
        nanList = [] #which columns in the row have NaN values
        for j, value in enumerate(row):
            if str(value.item()) == 'nan':
                nanList.append(colSequence[j])
        return nanList

def colSpecificActions( prevnextIndexRow, indexNanSet, colSequence ):
        prevnextIndexSet = set(nanList( prevnextIndexRow, colSequence ))
        # print("prevIndexNanSet:", prevIndexNanSet)
        sharedNanElements = indexNanSet.intersection(prevnextIndexSet)
        # print("sharedNanElements", sharedNanElements)

        colSpecific = [] #if passed empty allows no col specific actions to take place
        # if 2 in sharedNanElements: #specific action for temp col (can more easily work this out with bigger gaps in data)
        #     # print("Col specific action (temperature): ", colSequence[2])
        #     colSpecific.append(('temperature', 2))
        # if 33 in sharedNanElements: #specific action for platelets col
        #     # print("Col specific action (platelets): ", colSequence[33])
        #     colSpecific.append(('platelets', 33))
        # if 26 in sharedNanElements: #specific action for bilirubin_total col
        #     # print("Col specific action (bilirubin_total): ", colSequence[26])
        #     colSpecific.append(('bilirubin_total', 26))
        # if 19 in sharedNanElements: #specific action for creatinine col
        #     # print("Col specific action (creatinine): ", colSequence[19])
        #     colSpecific.append(('creatinine', 19))
        # print("sharedNanElements", sharedNanElements, "Col Specific actions?:", colSpecific)

        return sharedNanElements, colSpecific
