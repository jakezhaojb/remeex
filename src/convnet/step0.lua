--[[
First feedforward Convnet
By Jake
--]]

require 'torch'
require 'cunn'
require 'math'
require 'image'
require 'aux'
require 'osecond'
require 'xlua'
require 'model'

------------------------ OPTIONS -----------------------------

cmd = torch.CmdLine()
-- general
cmd:option('--devid', 2, 'GPU id')
cmd:option('--numthreads', 2, 'Number of CPU threads')
cmd:option('--seed', 1, 'Torch initial random seed')
cmd:option('--dataset', 'osecond', 'Dataset [osecond]')
-- model
cmd:option('--nPlanes', '256-512-512', 'Number of planes, eg. 16-32-16')
cmd:option('--kSizes', '3-3-3', 'Kernel sizes, eg. 9-7-5')
cmd:option('--stride', '1-1-1', 'stride, eg. 128-64-32')
cmd:option('--poolSizes', '2-0-2', 'Pooling Sizes, eg. 2-0-2')
-- training
cmd:option('--batchsize', 64, 'Minibatch size')
cmd:option('--nepoches', 100, 'Number of "eopches"')
cmd:option('--epochsize', 1000, 'Number of samples per "epoch"')
cmd:option('--lr', 0.02, 'Learning rate')
cmd:option('--lrd', 5e-5, 'Learning rate decay')
cmd:option('--mom', 0.9, 'momentum, eg. 0.9')
opt = cmd:parse(arg)

torch.manualSeed(opt.seed)
torch.setnumthreads(opt.numthreads)
cutorch.setDevice(opt.devid)
torch.setdefaulttensortype('torch.FloatTensor')

dataset = opt.dataset

params = {batchSize = opt.batchsize} -- TODO
datasource = OneSecondDatasource(params)
opt.inputSize = 22016
opt.labelSize = 172

-- get the model name
local irrelevant = {['nepoches'] = true,
		    ['devid'] = true,
		    ['epochsize'] = true,
		    ['numthreads'] = true,
          ['lrd'] = true}
local modelname = 'model'
for k, v in pairs(opt) do
   if not irrelevant[k] then
      local v2 = ''
      if type(v) == 'table' then
	 for i = 1, #v2 do
	    if i ~= 1 then
	       v2 = v2 .. '-'
	    end
	    v2 = v2 .. i
	 end
      elseif type(v) == 'boolean' then
	 if v then
	    v2 = 'T'
	 else
	    v2 = 'F'
	 end
      else
	 v2 = v
      end
      modelname = modelname .. '__' .. k .. '=' .. v2
   end
end
print(modelname)

-- convert opt to usable structures
local nplanes = convert_option(opt.nPlanes)
opt.nPlanes = {}
opt.nPlanes[1] = 1
for i = 1, #nplanes do
   opt.nPlanes[i+1] = nplanes[i]
end
opt.kSizes = convert_option(opt.kSizes)
opt.stride = convert_option(opt.stride)
opt.poolSizes = convert_option(opt.poolSizes)
opt.planeSizes = {opt.inputSize}
-- pool
for i = 1, #opt.poolSizes do
   local size = opt.planeSizes[i]
   if opt.poolSizes[i] ~= 0 then
      opt.planeSizes[i+1] = math.floor(size/opt.poolSizes[i])
   else
      opt.planeSizes[i+1] = size
   end
end
-------------------------- MODEL --------------------------------------
model = nn.Sequential()
for i = 1, #opt.kSizes do
   if opt.poolSizes[i] ~= 0 then
      model:add(get_conv_pool(opt.nPlanes[i], opt.nPlanes[i+1], opt.kSizes[i], opt.stride[i], opt.poolSizes[i]))
   else
      model:add(get_conv(opt.nPlanes[i], opt.nPlanes[i+1], opt.kSizes[i]), opt.stride[i])
   end
end
model:add(get_softmax_dropout(opt.labelSize, opt.nPlanes[#opt.nPlanes], opt.planeSizes[#opt.planeSizes], 0.5, {512, 128}))
print(model)

-- criterions
criterion = nn.MSECriterion()
criterion.sizeAverage = true

-- cuda
model:cuda()
criterion:cuda()
---------------------------------- TRAIN ------------------------------------

-- train !
local windows = {}
local k = 1
parameters, gradParameters = model:getParameters() 
for iEpoch = 1, opt.nepoches do
   model:training()
   local NLL_err = 0
   cutorch.synchronize()
   local tt = torch.Timer()
   for iIter = 1, opt.epochsize do
      model:zeroGradParameters()
      local data = datasource:nextBatch(opt.batchsize, 'train')
      local x = data[1]:cuda()
      local label = data[2]:cuda()
      local y = model:forward(x)
      local loss = criterion:forward(y, label)
      NLL_err = NLL_err + loss
      local dre_do = criterion:backward(y, label)
      model:backward(x, dre_do)
      -- Momentum added
      if opt.mom ~= 0 then
         if not state_dfdx then
            state_dfdx = torch.Tensor():typeAs(gradParameters):resize(gradParameters:size()):copy(gradParameters)
         else
            state_dfdx:mul(opt.mom):add(1-opt.mom, gradParameters)  -- TODO dampen
         end
         gradParameters:add(opt.mom, state_dfdx)
      end
      model:updateParameters(opt.lr / (1 + k * opt.lrd))
      k = k + 1
      xlua.progress(iIter, opt.epochsize)
   end
   cutorch.synchronize()
   print('Total time per iteration ' .. tt:time()['real'] / opt.epochsize)
   NLL_err = NLL_err / opt.epochsize
   print(iEpoch, 'NLL=' .. NLL_err)

   if iEpoch % 10 == 0 then      
      -- TODO evaluation code.
      os.execute('mkdir -p /scratch/jz1672/remeex/models/' .. modelname)
      torch.save('/scratch/jz1672/remeex/models/' .. modelname .. '/epoch_' .. iEpoch .. '.t7b', {model=model, opt=opt}, 'binary') -- TODO: warnings
   end
   collectgarbage()
end
