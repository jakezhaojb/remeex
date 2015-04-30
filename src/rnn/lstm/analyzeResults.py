import os
import pandas as pd
import matplotlib.pyplot as plt
import prettyplotlib as ppl
import datetime
import time


#-------------------------------------------------------------
tag = 'experiment1_voicing_01'
#-------------------------------------------------------------

path = os.path.join(os.path.join('../results',tag),'results')
train = {}
valid = {}
test = {}
times = {}

def generateName(mydict,name):
    # if name is already in dict, then append a number to it
    # e.g. results1, results2, etc. 
    if name in mydict.keys():
        j=1
        name = name+"_"+str(j)
        while name in mydict.keys():
            j=j+1
            name = name[:-1]+str(j)
    return name

valid_dirtimes = []
for rundir in os.listdir(path): 
    train_results = os.path.join(path,rundir,'train.log')
    valid_results = os.path.join(path,rundir,'validation.log')
    test_results = os.path.join(path,rundir,'test.log')
    time_results = os.path.join(path,rundir,'times.log')
  
    if os.path.isdir(os.path.join(path,rundir)):
        valid_dirtimes.append([rundir,time.time() - os.stat(os.path.join(path,rundir)).st_ctime])
    if os.path.isfile(train_results):
        name = generateName(train,rundir)
        try:
            train[name] = pd.read_csv(train_results)
        except:
            pass
    if os.path.isfile(valid_results):
        name = generateName(valid,rundir)
        try:
            valid[name] = pd.read_csv(valid_results)
        except:
            pass
    if os.path.isfile(test_results):
        name = generateName(test,rundir)
        try:
            test[name] = pd.read_csv(test_results)
        except:
            pass
    if os.path.isfile(time_results):
        name = generateName(times,rundir)
        try:
            times[name] = pd.read_csv(time_results,header=None)
        except:
            pass
    

trainPlot = sorted([[k,v] for k,v in train.iteritems()],key=lambda x:x[0])
validPlot = sorted([[k,v] for k,v in valid.iteritems()],key=lambda x:x[0])
#validPlot = sorted([[k,v] for k,v in newvalids.iteritems()],key=lambda x:x[0])
testPlot = sorted([[k,v] for k,v in test.iteritems()],key=lambda x:x[0])
timesPlot = sorted([[k,v/60] for k,v in times.iteritems()],key=lambda x:x[0])

train_bestval = {k:max(max(v.values)) for [k,v] in trainPlot}
valid_bestval = {k:max(max(v.values)) for [k,v] in validPlot}
test_bestval = {k:v.iloc[-1,0] for [k,v] in testPlot}

exclusions=[]
#exclusions=['128c_256c_256c_512f_dr50']

def pltresults(results,title,tag,exclusions=[],bestval=None,fromiter=0,legend=True):
    if len(results) > 0:
        fig,ax = plt.subplots()       
        fig.set_size_inches(10,5)
        for k,v in results:
            if k not in exclusions:
                v = v.iloc[min(fromiter,len(v)):]
                its = len(v)
                if bestval:
                    mybestval = bestval[k]
                else:
                    mybestval = ""
                label = k + " - " + str(mybestval) + "% - " + str(its) + " iters"
                ppl.plot(ax,range(len(v)),v,label=label)
                if legend: ax.legend(loc='lower right')
                ax.set_title(title + ' - ' + tag)
                print label
        plt.show()
    else:
        print "no results for " + title
        
def pltdelta(results,title,tag,exclusions=[],bestval=None,fromiter=0):
    if len(results) > 0:
        fig,ax = plt.subplots()       
        fig.set_size_inches(10,5)
        for k,v in results:
            if k not in exclusions:
                minit = min(fromiter,len(v)-1)
                maxit = len(v)
                v = v.iloc[minit:]
                its = len(v)
                if bestval:
                    mybestval = bestval[k]
                else:
                    mybestval = ""
                label = k + " - " + str(mybestval) + "% - " + str(its) + " iters"
                ppl.plot(ax,range(minit,maxit),v-v.iloc[0].values,label=label)
                #ax.legend(loc='u   pper right')
                ax.set_title(title + ' - delta - ' + tag)
                print label
        plt.show()
    else:
        print "no results for " + title

pltresults(trainPlot,'train',tag,exclusions,train_bestval)
pltresults(validPlot,'validation',tag,exclusions,valid_bestval)
#pltresults(testPlot,'test',tag,exclusions,test_bestval)
#pltresults(testPlot,'test',tag,exclusions,test_bestval,fromiter=40,legend=False)
#pltdelta(testPlot,'test',tag,exclusions,test_bestval,fromiter=124)
#pltresults(timesPlot,'times',tag,exclusions)

print "avg validation error =",mean(valid_bestval.values())
print "avg test error =",mean(test_bestval.values())


'''
fig,ax = plt.subplots()       
fig.set_size_inches(10,5)
for k,v in timesPlot:
    if k in valid.keys():
        n = min(len(v),len(valid[k]))
        mybestval = valid_bestval[k]
        label = k + " - " + str(mybestval) + "% - " + str(n) + " iters"
        ppl.plot(ax,v.iloc[:n]/60,valid[k].iloc[:n],label=label)
        ax.legend(loc='lower right')
        ax.set_xlabel("minutes")
        ax.set_ylabel("% accuracy")
        ax.set_title('times' + ' - ' + tag)
        print label
plt.show()
'''

