
require 'torch'
require 'audio'
require 'csvigo'
require 'nn'
require 'optim'
--require('mobdebug').start()

torch.setdefaulttensortype('torch.FloatTensor')

-- load audio
X = audio.load('LizNelson_Rainfall_MIX.wav'):transpose(1,2):float()

-- average the left and right audio
X = (X[{{},1}] + X[{{},2}])/2

-- load the melody csv
query = csvigo.load{path='melody.csv',mode='query'}
melody = query('all')
melody.size = function() return #melody['end_ind'] end

i = 0
y = torch.Tensor(melody:size()):apply(function() i=i+1 return tonumber(melody['label'][i]) end):float()

-- define window for convnet
windowLeft = 512
windowRight = 512
melodyWindow = 256
window = melodyWindow + windowLeft + windowRight

ind_start = math.ceil(windowLeft/melodyWindow)
ind_end = melody:size() - math.ceil(windowRight/melodyWindow)
num_samples = ind_end - ind_start


model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(1,16,64,1,8)) -- 1280->159  
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(3,1,3,1)) -- 159->53

fully_connected_size = 16*53
model:add(nn.View(fully_connected_size))
model:add(nn.Linear(fully_connected_size, 16))
model:add(nn.ReLU())
model:add(nn.Linear(16, 2))

print(model)

model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

classes = {'0','1'}
confusion = optim.ConfusionMatrix(classes)

parameters,gradParameters = model:getParameters()

optimState = {
  learningRate = 1e-3,
  weightDecay = 0,
  learningRateDecay = 1e-7,
  momentum = 0.95,
}
optimMethod = optim.sgd

function train()
    collectgarbage()

	-- epoch tracker
	epoch = epoch or 1

	-- local vars
	local time = sys.clock()

	-- set model to training mode (for modules that differ in training and testing, like Dropout)
	model:training()

	-- shuffle at each epoch
	shuffle = torch.randperm(num_samples)

	-- do one epoch
	print('------------------------------TRAINING--------------------------------------')
	print('optim state:')
	print(optimState)
	print("==> online epoch # " .. epoch .. ' [batchSize = 1]')
	for t = 1,num_samples do
		-- disp progress
		collectgarbage()
		xlua.progress(t, num_samples)

		local i = shuffle[t]+ind_start
		local X_start = tonumber(melody['start_ind'][i])-windowLeft
		local X_end = tonumber(melody['end_ind'][i])+windowRight
		inputs = X[{{X_start,X_end-1}}]:reshape(X_end-X_start,1,1)
		targets = y[{{i}}]

		local feval = function(x)
	      	if x ~= parameters then
				parameters:copy(x)
			end

			-- reset gradients
			gradParameters:zero()

			-- f is the average of all criterions
			local f = 0

			-- evaluate function for complete mini batch
			-- estimate f
			print(inputs:type())
			print(targets:type())
			local output = model:forward(inputs)
			local err = criterion:forward(output, targets)
			f = f + err

			-- estimate df/dW
			local df_do = criterion:backward(output, targets)
			model:backward(inputs, df_do)

			-- update confusion
			confusion:batchAdd(output, targets)

			-- return f and df/dX
			return f,gradParameters:float()
		end
		optimMethod(feval, parameters:float(), optimState)
	end

	-- time taken
	time = sys.clock() - time
	time = time / trainData:size()
	print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

	-- print confusion matrix
	print(confusion)

	-- next epoch
	confusion:zero()
	epoch = epoch + 1
end

--train()
--while true do
--	train()
--end
