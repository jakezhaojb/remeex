import hmm
import hmm_predict
import os


run_name = 'hmm3_voicing_detection'

melodies_folderpath = '/home/jmj418/melody_type1'
cqt_folderpath = '/home/jmj418/cqt2'
datasplits_filepath = '../data/datasplits.txt'
models_dir = '../../models/hmm'
output_dir = os.path.join('/home/jmj418/hmm_predictions',run_name)

filename='%s.pk1' % run_name

hmm.generate_hmm_params(binary=True,
						filename=filename,
						melodies_folderpath = melodies_folderpath,
						cqt_folderpath = cqt_folderpath,
						datasplits_filepath = datasplits_filepath,
						models_dir = models_dir)



hmm_predict.hmm_predict(filename=filename,
		                melodies_folderpath = melodies_folderpath,
		                cqt_folderpath = cqt_folderpath,
		                datasplits_filepath = datasplits_filepath,
		                models_dir = models_dir,
		                output_dir = output_dir)