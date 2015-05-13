rootdir = '../results'
experiment_name = 'experiment2_voicing'
resume = false
device = 2
trainParams = {}
trainParams[2] = {
	numEpochs = 50,
	batchSize = 1,
	learningRate = 1e-3,
	learningRateDecay = 1e-7,
	optimization = "SGD"
}
maxNumSamples = 10000

print('==> switching to CUDA')
require 'cunn'
cutorch.setDevice(device)


function formatnum(num)
	str = tostring(num)
	while string.len(str) < 2 do
		str = '0' .. str
	end
	return str
end

if resume==false then
	status = false
	num = 1
	experiments = ""
	while status ~= true and num < 10 do
		tag = experiment_name .. '_' .. formatnum(num)
		experiments = paths.concat(rootdir, tag)
		response = os.rename(experiments,experiments)
		status = response ~= true
		num = num+1
		print(status)
	end
	sys.execute('mkdir -p ' .. experiments)

	sys.execute('cp *.lua ' .. experiments)

	results = paths.concat(experiments, 'results')
	sys.execute('mkdir -p ' .. results)
	path = experiments
	save = results
else
	path = './'
	save = 'results'
end




params = {}
params['p64c55_p128c65P_512f_64f_dr70_k11'] = {['model']='p64c55_p128c65P_512f_64f',['dropout']=0.7,['kL']=5,['kR']=5}
params['p64c55_p128c65P_512f_64f_dr70_k21'] = {['model']='p64c55_p128c65P_512f_64f',['dropout']=0.7,['kL']=10,['kR']=10}
params['p64c55_p128c65P_512f_64f_dr70_k41'] = {['model']='p64c55_p128c65P_512f_64f',['dropout']=0.7,['kL']=20,['kR']=20}

params['p64c55_p128c65_512f_64f_dr70_k21'] = {['model']='p64c55_p128c65_512f_64f',['dropout']=0.7,['kL']=10,['kR']=10}
params['p32c55_p64c65_256f_32f_dr70_k21'] = {['model']='p32c55_p64c65_256f_32f',['dropout']=0.7,['kL']=10,['kR']=10}



opt = {}
-- global:
opt['seed'] = 1 -- fixed input seed for repeatable experiments
opt['threads'] = 8 -- number of threads
opt['device'] = device -- number of threads
opt['path'] = path -- location of source files
opt['binarylabel'] = true -- detect voicing=true, detect melody=false
-- data:
opt['maxNumSamples'] = maxNumSamples
opt['preprocessedData'] = '/home/justin/cqt/t7b/' -- where the preprocessed data lives
opt['loadDataFile'] = '1_data.lua' -- file used to load data
-- loss:
opt['loss'] = 'nll' -- type of loss function to minimize: nll | mse | margin
-- training:
opt['trainFile'] = '4_train.lua'
opt['save'] = save -- subdirectory to save/log experiments in
opt['plot'] = true -- live plot
opt['optimization'] = 'ADADELTA' -- optimization method: SGD | ASGD | CG | LBFGS
opt['model'] = 'cuda1' -- type of model to construct: linear | mlp | convnet
opt['dropout'] = 0.7 -- max number of epochs to run
opt['trainParams'] = trainParams
opt['learningRate'] = 1e-3 -- learning rate at t= 0
opt['learningeRateDecay'] = 1e-7 -- learning rate decay
opt['batchSize'] = batchSize -- mini-batch size (1 =  pure stochastic)
opt['initbatchSize'] = initbatchSize -- begin with minbatchsize, then switch to batchsize at t0
opt['batchLearningRate'] = 1e-4 -- begin with minbatchsize, then switch to batchsize at t0
opt['weightDecay'] = 0 -- weight decay (SGD only)
opt['momentum'] = 0.95 -- momentum (SGD only)
opt['t0'] = t0 -- start averaging at t0 (ASGD only), in nb of epochs
opt['lambda'] = 1e-4 -- ASGD lambda
opt['type'] = 'cuda' -- type: double | float | cuda
opt['maxEpochs'] = maxEpochs -- max number of epochs to run
-- test:
opt['runValidation'] = true -- toggle validation
opt['runTest'] = false -- toggle test


for key,optlist in pairs(params) do
	opt['tag'] = key -- 'name of run, used as name for results folder')
	for optname,optval in pairs(optlist) do 
		opt[optname] = optval
	end

	epoch = 1
	if resume then
		dofile 'resume.lua'
	else
		dofile (paths.concat(experiments, 'doall.lua'))
	end
end















