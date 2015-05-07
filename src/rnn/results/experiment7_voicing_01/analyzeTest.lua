----------------------------------------------------------------------
-- This tutorial shows how to train different models on the street
-- view house number dataset (SVHN),
-- using multiple optimization techniques (SGD, ASGD, CG), and
-- multiple types of models.
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem. 
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------
require 'lfs'
require 'torch'
require 'optim'
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

----------------------------------------------------------------------
--dofile (paths.concat(opt.path, '5_test.lua'))

----------------------------------------------------------------------

testData = torch.load(paths.concat('../../data/stl10_binary',"testData8000_normalized.bin"))
tesize = testData.data:size()[1]
testData.data = testData.data:reshape(tesize,3,96,96)
testData.size = function() return tesize end


classes = {'1','2','3','4','5','6','7','8','9','0'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)


experiments = {}

for file in lfs.dir('./results/') do
	if file ~= '.' and file ~= '..' then
		table.insert(experiments,file)
	end
end
bestaccuracy = 1
for i=1,#experiments do
	print(experiments[i])
	path = paths.concat(paths.concat('results',experiments[i]),'model_best.net')
	if not pcall(function() model = torch.load(path) end) then
		print(experiments[i] .. ': model_best.net could not load')
	else
		dofile '5_test.lua'
		if not pcall(function() test() end) then
			print(experiments[i] .. ': test() failed')
		end
	end
end


