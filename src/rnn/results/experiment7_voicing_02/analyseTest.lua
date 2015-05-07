----------------------------------------------------------------------
----------------------------------------------------------------------

----------------------------------------------------------------------
print '==> processing options'

if not opt then
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('SVHN Loss Function')
	cmd:text()
	cmd:text('Options:')
	-- global:
	cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
	cmd:option('-threads', 8, 'number of threads')
	cmd:option('-device', 1, 'number of threads')
	cmd:option('-path', './', 'location of source files')
	-- data:
	cmd:option('-data', '../../data/stl10_binary', 'subdirectory containing data files')
	-- model:
	cmd:option('-model', 'cuda1', 'type of model to construct: linear | mlp | convnet')
	-- loss:
	cmd:option('-loss', 'mse', 'type of loss function to minimize: nll | mse | margin')
	-- training:
	cmd:option('-save', '../../results', 'subdirectory to save/log experiments in')
	cmd:option('-tag', 'default', 'name of run, used as name for results folder')
	cmd:option('-plot', true, 'live plot')
	cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
	cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
	cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
	cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
	cmd:option('-momentum', 0.9, 'momentum (SGD only)')
	cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
	cmd:option('-lambda', 1e-4, 'ASGD lambda')
	cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
	cmd:option('-type', 'cuda', 'type: double | float | cuda')
	cmd:option('-maxEpochs', 100, 'max number of epochs to run')
	cmd:option('-dropout', 0.5, 'max number of epochs to run')
	cmd:text()
	opt = cmd:parse(arg or {})
end
	

tag = opt.tag
folder_tag = tag
results_folder = paths.concat(opt.save,folder_tag)
sys.execute('mkdir -p ' .. results_folder)

function doall()
	-- save out params
	params = io.open(paths.concat(results_folder,'params.txt'),'a')
	params:write('parameter, value','\n')
	params:write('data, ' .. tostring(opt.data),'\n')
	params:write('model, ' .. tostring(opt.model),'\n')
	params:write('optimization, ' .. tostring(opt.optimization),'\n')
	params:write('learningRate, ' .. tostring(opt.learningRate),'\n')
	params:write('batchSize, ' .. tostring(opt.batchSize),'\n')
	params:write('weightDecay, ' .. tostring(opt.weightDecay),'\n')
	params:write('momentum, ' .. tostring(opt.momentum),'\n')
	params:write('t0, ' .. tostring(opt.t0),'\n')
	params:write('maxIter, ' .. tostring(opt.maxIter),'\n')
	params:write('maxEpochs, ' .. tostring(opt.maxEpochs),'\n')
	params:flush()
	params:close()

	batchSize = opt.batchSize
	opt.batchSize = 1


	-- nb of threads and fixed seed (for repeatable experiments)
	if opt.type == 'float' then
	   print('==> switching to floats')
	   torch.setdefaulttensortype('torch.FloatTensor')
	elseif opt.type == 'cuda' then
	   print('==> switching to CUDA')
	   require 'cunn'
	   cutorch.setDevice(opt.device)
	   torch.setdefaulttensortype('torch.FloatTensor')
	end
	torch.setnumthreads(opt.threads)
	torch.manualSeed(opt.seed)



	----------------------------------------------------------------------
	print '==> executing all'

	--if not data_load_complete then dofile (paths.concat(opt.path, '1_data.lua')) end
	dofile (paths.concat(opt.path, '1_data.lua'))
	dofile (paths.concat(opt.path, '2_model.lua'))
	dofile (paths.concat(opt.path, '3_loss.lua'))
	dofile (paths.concat(opt.path, '4_train.lua'))
	dofile (paths.concat(opt.path, '5_test.lua'))

	----------------------------------------------------------------------
	print '==> training!'

	p = 0
	while p<opt.maxEpochs do
		if p == opt.t0 then
			opt.batchSize = batchSize
			opt.learningRate = opt.learningRate/10
			dofile (paths.concat(opt.path, '4_train.lua'))
		end
		p = p+1
		train()
		test()
	end
end

