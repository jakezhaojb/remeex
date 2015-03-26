
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
--model:add(nn.ReLU())
--model:add(nn.SpatialMaxPooling(3,1,3,1)) -- 159->53

--fully_connected_size = 16*53
--model:add(nn.View(fully_connected_size))
--model:add(nn.Linear(fully_connected_size, 16))
--model:add(nn.ReLU())
--model:add(nn.Linear(16, 2))

print(model)

--model:add(nn.LogSoftMax())
--criterion = nn.ClassNLLCriterion()

parameters,gradParameters = model:getParameters()

i = 1+ind_start
X_start = tonumber(melody['start_ind'][i])-windowLeft
X_end = tonumber(melody['end_ind'][i])+windowRight
inputs = X[{{X_start,X_end-1}}]:reshape(X_end-X_start,1,1)
targets = y[{{i}}]

--output = model:forward(inputs)

--err = criterion:forward(output, targets)
