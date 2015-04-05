latest version: 
* hmm2.pk1 = tuple(A,means,covmats,num_samples)

older versions:

* hmm.pk1 = tuple(A,means,covmats)

load using the following commands:


	import os
	import pickle


	models_dir = '../../models/hmm'

	def pickleLoad(inputName):
	    pk1_file = open(inputName+'.pk1', 'rb')
	    pyObj = pickle.load(pk1_file)
	    return pyObj

	A,means,covmats,num_samples = pickleLoad(os.path.join(models_dir,'hmm2'))

accuracy files:
* accuracies2.pk1 = tuple(train_accuracies,train_voice_det_confusion,validation_accuracies,validation_voice_det_confusion)
