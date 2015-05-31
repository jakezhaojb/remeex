import numpy as np
import matplotlib.pyplot as plt
import prettyplotlib as ppl
import seaborn

tags = []
tags.append('test_LSTM_melody')
tags.append('test_LSTM_melody_3l')
    
weights = [1,1]

CORRECT = []
WRONGPITCH = []
FNR = []
VDA = []

for tag in tags:
    confusion = np.loadtxt('results/%s/confusion.csv' % tag,delimiter=',')

    total = np.sum(confusion[1:,:],axis=1)
    correct = np.sum(confusion[1:,1:]*np.eye(84),axis=1)/total*100
    wrongpitch = np.sum(confusion[1:,1:]*(1-np.eye(84)),axis=1)/total*100
    fnr = confusion[1:,0]/total*100
    
    ignore = np.sum(confusion,axis=1) == 0
    
    confusion_ = np.copy(confusion[ignore==False])
    confusion_[0,0] = 0
    plt.pcolor(confusion_);
    plt.gca().invert_yaxis()
    plt.title(tag)
    plt.show()
    
    ind = np.arange(84)
    
    fig, ax = plt.subplots(1)
    plt.bar(ind,correct,label='correct pitch')
    plt.bar(ind,wrongpitch,color='r',bottom=correct,label='wrong pitch')
    plt.bar(ind,fnr,color='y',bottom=wrongpitch+correct,label='false negative')
    legend = plt.legend(frameon = 1,bbox_to_anchor=(0.5,0.85))
    frame = legend.get_frame()
    frame.set_color('white')
    plt.title('%s error analysis' % tag)
    plt.xlabel('notes')
    plt.ylabel('%')
    plt.ylim(0,100)
    plt.show()

    total = np.sum(confusion[1:,:])*1.0
    CORRECT.append(np.sum(confusion[1:,1:]*np.eye(84))/total*100)
    WRONGPITCH.append(np.sum(confusion[1:,1:]*(1-np.eye(84)))/total*100)
    FNR.append(np.sum(confusion[1:,0])/total*100)
    VDA.append((confusion[0,0]+np.sum(confusion[1:,1:]))*1.0/np.sum(confusion))

correct = np.array(CORRECT)
wrongpitch = np.array(WRONGPITCH)
fnr = np.array(FNR)

fig, ax = plt.subplots(1)
ind = np.arange(len(tags))
plt.bar(ind,correct,label='correct pitch')
plt.bar(ind,wrongpitch,color='r',bottom=correct,label='wrong pitch')
plt.bar(ind,FNR,color='y',bottom=wrongpitch+correct,label='false negative')
legend = plt.legend(frameon = 1)
frame = legend.get_frame()
frame.set_color('white')
plt.title('weighted negative log likelihood\nmelody extraction error analysis')
plt.ylabel('%')
plt.xlabel('no melody weight')
plt.ylim(0,100)
ax.set_xticks(ind+.4)
ax.set_xticklabels(weights)
plt.show()


for i,tag in enumerate(tags):
    print tag, VDA[i]