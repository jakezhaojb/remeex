
require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
require 'nngraph'


----------------------------------------------------------------------
print '==> define parameters'

model = {}

-- 10-class problem
noutputs = 2

-- input dimensions
nfeats = 300
nwords = opt.nWords
width = 32
height = 32
ninputs = nfeats*width*height

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2

-- hidden units, filter sizes (for ConvNet only):
nstates = {64,64,128}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)

----------------------------------------------------------------------
print '==> construct model'

function lstm(i, prev_c, prev_h)
   local function new_input_sum()
      local i2h            = nn.Linear(opt.rnn_size, opt.rnn_size)
      local h2h            = nn.Linear(opt.rnn_size, opt.rnn_size)
      return nn.CAddTable()({i2h(i), h2h(prev_h)})
   end
   local function new_input_sum_peepholes()
      -- add connections from memory cells to input, output, and forget gates
      local i2h            = nn.Linear(opt.rnn_size, opt.rnn_size)
      local h2h            = nn.Linear(opt.rnn_size, opt.rnn_size)
      local c2h            = nn.Linear(opt.rnn_size, opt.rnn_size)
      return nn.CAddTable()({i2h(i), h2h(prev_h), c2h(prev_c)})
   end
   if opt.peepholes ~= true then
      new_input_sum_peepholes = new_input_sum
   end
   local in_gate          = nn.Sigmoid()(new_input_sum_peepholes())
   local forget_gate      = nn.Sigmoid()(new_input_sum_peepholes())
   local in_gate2         = nn.Tanh()(new_input_sum())
   local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_gate2})
   })
   local out_gate         = nn.Sigmoid()(new_input_sum_peepholes())
   local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
   return next_c, next_h
end

function transfer_data(x)
   return x:cuda()
end

function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = nn.Linear(opt.num_inputs,opt.rnn_size)(x)}
  local next_s           = {}
  local split         = {prev_s:split(2 * opt.layers)}
  for layer_idx = 1, opt.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(opt.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(opt.rnn_size, opt.num_outputs)
  local dropped          = nn.Dropout(opt.dropout)(i[opt.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = nn.ClassNLLCriterion()({pred, y})
  -- setup predictions as additional output to model
  local network           = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s), nn.Identity()(pred)})
  network:getParameters():uniform(-opt.init_weight, opt.init_weight)
  return transfer_data(network)
end

function g_cloneManyTimes(net, T)
  local clones = {}
  local params, gradParams = net:parameters()
  if params == nil then
    params = {}
  end
  local paramsNoGrad
  if net.parametersNoGrad then
    paramsNoGrad = net:parametersNoGrad()
  end
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)
  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    local cloneParamsNoGrad
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
    if paramsNoGrad then
      cloneParamsNoGrad = clone:parametersNoGrad()
      for i =1,#paramsNoGrad do
        cloneParamsNoGrad[i]:set(paramsNoGrad[i])
      end
    end
    clones[t] = clone
    collectgarbage()
  end
  mem:close()
  return clones
end


if opt.model == 'LSTM' then


  print("Creating RNN LSTM network.")
  local core_network = create_network()
  parameters, gradParameters = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, opt.kL+1 do
    model.s[j] = {}
    for d = 1, 2 * opt.layers do
      model.s[j][d] = transfer_data(torch.zeros(batchSize, opt.rnn_size))
    end
  end
  for d = 1, 2 * opt.layers do
    model.start_s[d] = transfer_data(torch.zeros(batchSize, opt.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(batchSize, opt.rnn_size))
  end
  model.core_network = core_network
  model.rnnL = g_cloneManyTimes(core_network, opt.kL)
  model.rnnt = g_cloneManyTimes(core_network, 1)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(opt.kL+1))
  -- store parameters and perplexities
  model.params = opt
  model.accuracy = {}
  model.accuracy.train = {}
  model.accuracy.validation = {}
  model.accuracy.test = {}

elseif opt.model == 'LSTM2' then

  print("Creating RNN LSTM network.")
  local core_network = create_network()
  parameters, gradParameters = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, opt.kL do
    model.s[j] = {}
    for d = 1, 2 * opt.layers do
      model.s[j][d] = transfer_data(torch.zeros(batchSize, opt.rnn_size))
    end
  end
  for d = 1, 2 * opt.layers do
    model.start_s[d] = transfer_data(torch.zeros(batchSize, opt.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(batchSize, opt.rnn_size))
  end
  model.core_network = core_network
  model.rnnL = g_cloneManyTimes(core_network, opt.kL)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(opt.kL))
  -- store parameters and perplexities
  model.params = opt
  model.accuracy = {}
  model.accuracy.train = {}
  model.accuracy.validation = {}
  model.accuracy.test = {}

elseif opt.model == 'T_p168c5_p336c6P_p336c3P_512f_64f' then
   -- a typical modern convolution network (conv+relu+pool)
   model = nn.Sequential()

   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   local width = opt.kL+opt.kR+1
   model:add(nn.SpatialZeroPadding(0,0,2,2)) -- 21x84 -> 25x88
   model:add(nn.TemporalConvolution(84, 168, 5, 1)) -- 25x88 -> 21x84
   model:add(nn.ReLU())

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(0,0,2,2)) -- 21x84 -> 25x88
   model:add(nn.TemporalConvolution(168, 336, 6, 1)) -- 25x88 -> 20x84
   model:add(nn.TemporalMaxPooling(2,2)) -- 20x84 -> 10x84
   model:add(nn.ReLU())
   width = math.floor((width-1)/2)

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(0,0,1,1)) -- 10->12
   model:add(nn.TemporalConvolution(336, 336, 3, 1)) -- 12->10
   model:add(nn.TemporalMaxPooling(2,2)) -- 10->5
   model:add(nn.ReLU())
   width = math.floor((width)/2)

   -- stage 3 : standard 2-layer neural network
   fully_connected_size = 336*width
   model:add(nn.View(fully_connected_size))
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(fully_connected_size, 512))
   model:add(nn.ReLU())
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(512, 64))
   model:add(nn.ReLU())
   model:add(nn.Linear(64, noutputs))

elseif opt.model == 'T_p64c5_p32c6P_64f_16f' then
   -- a typical modern convolution network (conv+relu+pool)
   model = nn.Sequential()

   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   local width = opt.kL+opt.kR+1
   model:add(nn.SpatialZeroPadding(0,0,2,2)) -- 21x84 -> 25x88
   model:add(nn.TemporalConvolution(84, 64, 5, 1)) -- 25x88 -> 21x84
   model:add(nn.ReLU())

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(0,0,2,2)) -- 21x84 -> 25x88
   model:add(nn.TemporalConvolution(64, 32, 6, 1)) -- 25x88 -> 20x84
   model:add(nn.TemporalMaxPooling(2,2)) -- 20x84 -> 10x84
   model:add(nn.ReLU())
   width = math.floor((width-1)/2)

   -- stage 3 : standard 2-layer neural network
   fully_connected_size = 32*width
   model:add(nn.View(fully_connected_size))
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(fully_connected_size, 64))
   model:add(nn.ReLU())
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(64, 16))
   model:add(nn.ReLU())
   model:add(nn.Linear(16, noutputs))


elseif opt.model == 'p32c55_p64c65_256f_32f' then
   -- a typical modern convolution network (conv+relu+pool)
   model = nn.Sequential()

   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   local width = opt.kL+opt.kR+1
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 21x84 -> 25x88
   model:add(nn.SpatialConvolutionMM(1, 32, 5, 5)) -- 25x88 -> 21x84
   model:add(nn.ReLU())

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 21x84 -> 25x88
   model:add(nn.SpatialConvolutionMM(32, 64, 5, 6)) -- 25x88 -> 20x84
   model:add(nn.ReLU())
   width = width - 1

   -- stage 3 : standard 2-layer neural network
   fully_connected_size = 64*width*84
   model:add(nn.View(fully_connected_size))
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(fully_connected_size, 256))
   model:add(nn.ReLU())
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(256, 32))
   model:add(nn.ReLU())
   model:add(nn.Linear(32, noutputs))

elseif opt.model == 'p64c55_p128c65_512f_64f' then
   -- a typical modern convolution network (conv+relu+pool)
   model = nn.Sequential()

   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   local width = opt.kL+opt.kR+1
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 21x84 -> 25x88
   model:add(nn.SpatialConvolutionMM(1, 64, 5, 5)) -- 25x88 -> 21x84
   model:add(nn.ReLU())

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 21x84 -> 25x88
   model:add(nn.SpatialConvolutionMM(64, 128, 5, 6)) -- 25x88 -> 20x84
   model:add(nn.ReLU())
   width = width - 1

   -- stage 3 : standard 2-layer neural network
   fully_connected_size = 128*width*42
   model:add(nn.View(fully_connected_size))
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(fully_connected_size, 512))
   model:add(nn.ReLU())
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(512, 64))
   model:add(nn.ReLU())
   model:add(nn.Linear(64, noutputs))

elseif opt.model == 'S_p64c55_p128c65P_512f_64f' then
   -- a typical modern convolution network (conv+relu+pool)
   model = nn.Sequential()

   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   local width = opt.kL+opt.kR+1
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 21x84 -> 25x88
   model:add(nn.SpatialConvolutionMM(1, 64, 5, 5)) -- 25x88 -> 21x84
   model:add(nn.ReLU())
   --model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 96->48

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 21x84 -> 25x88
   model:add(nn.SpatialConvolutionMM(64, 128, 5, 6)) -- 25x88 -> 20x84
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 10x42
   width=math.floor((width-1)/poolsize)

   -- stage 3 : standard 2-layer neural network
   fully_connected_size = 128*width*42
   model:add(nn.View(fully_connected_size))
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(fully_connected_size, 512))
   model:add(nn.ReLU())
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(512, 64))
   model:add(nn.ReLU())
   model:add(nn.Linear(64, noutputs))

elseif opt.model == 'S_p64c5x41_p128c6x5P_256f_32f' then
   -- a typical modern convolution network (conv+relu+pool)
   model = nn.Sequential()

   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   local width = opt.kL+opt.kR+1
   model:add(nn.SpatialZeroPadding(0,0,2,2)) -- 21x84 -> 25x84
   model:add(nn.SpatialConvolutionMM(1, 64, 41, 5)) -- 25x84 -> 21x44
   model:add(nn.ReLU())
   --model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 96->48

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 21x44 -> 25x48
   model:add(nn.SpatialConvolutionMM(64, 128, 5, 6)) -- 25x48 -> 20x44
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 20x44 -> 10x22
   width=math.floor((width-1)/poolsize)

   -- stage 3 : standard 2-layer neural network
   fully_connected_size = 128*width*22
   model:add(nn.View(fully_connected_size))
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(fully_connected_size, 256))
   model:add(nn.ReLU())
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(256, 32))
   model:add(nn.ReLU())
   model:add(nn.Linear(32, noutputs))

elseif opt.model == 'S_p64c5x21_p128c6x5P_256f_32f' then
   -- a typical modern convolution network (conv+relu+pool)
   model = nn.Sequential()

   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   local width = opt.kL+opt.kR+1
   model:add(nn.SpatialZeroPadding(0,0,2,2)) -- 21x84 -> 25x84
   model:add(nn.SpatialConvolutionMM(1, 64, 21, 5)) -- 25x84 -> 21x64
   model:add(nn.ReLU())
   --model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 96->48

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 21x64 -> 25x68
   model:add(nn.SpatialConvolutionMM(64, 128, 5, 6)) -- 25x68 -> 20x64
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 20x64 -> 10x32
   width=math.floor((width-1)/poolsize)

   -- stage 3 : standard 2-layer neural network
   fully_connected_size = 128*width*32
   model:add(nn.View(fully_connected_size))
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(fully_connected_size, 256))
   model:add(nn.ReLU())
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(256, 32))
   model:add(nn.ReLU())
   model:add(nn.Linear(32, noutputs))


elseif opt.model == 'S_p64c5x41_p64c6x5P_p64c5x5P_256f_32f' then
   -- a typical modern convolution network (conv+relu+pool)
   model = nn.Sequential()

   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   local width = opt.kL+opt.kR+1
   model:add(nn.SpatialZeroPadding(0,0,2,2)) -- 21x84 -> 25x84
   model:add(nn.SpatialConvolutionMM(1, 64, 41, 5)) -- 25x84 -> 21x44
   model:add(nn.ReLU())
   --model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 96->48

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 21x44 -> 25x48
   model:add(nn.SpatialConvolutionMM(64, 64, 5, 6)) -- 25x48 -> 20x44
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 20x44 -> 10x22
   width=math.floor((width-1)/poolsize)

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 10x22 -> 14x26
   model:add(nn.SpatialConvolutionMM(64, 64, 5, 5)) -- 14x26 -> 10x22
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 10x22 -> 5x11
   width=math.floor((width)/poolsize)

   -- stage 3 : standard 2-layer neural network
   fully_connected_size = 64*width*11
   model:add(nn.View(fully_connected_size))
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(fully_connected_size, 256))
   model:add(nn.ReLU())
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(256, 32))
   model:add(nn.ReLU())
   model:add(nn.Linear(32, noutputs))

elseif opt.model == 'S_p64c5x41_p128c6x5P_p128c5x5P_256f_32f' then
   -- a typical modern convolution network (conv+relu+pool)
   model = nn.Sequential()

   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   local width = opt.kL+opt.kR+1
   model:add(nn.SpatialZeroPadding(0,0,2,2)) -- 21x84 -> 25x84
   model:add(nn.SpatialConvolutionMM(1, 64, 41, 5)) -- 25x84 -> 21x44
   model:add(nn.ReLU())
   --model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 96->48

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 21x44 -> 25x48
   model:add(nn.SpatialConvolutionMM(64, 128, 5, 6)) -- 25x48 -> 20x44
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 20x44 -> 10x22
   width=math.floor((width-1)/poolsize)

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 10x22 -> 14x26
   model:add(nn.SpatialConvolutionMM(128, 128, 5, 5)) -- 14x26 -> 10x22
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 10x22 -> 5x11
   width=math.floor((width)/poolsize)

   -- stage 3 : standard 2-layer neural network
   fully_connected_size = 128*width*11
   model:add(nn.View(fully_connected_size))
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(fully_connected_size, 256))
   model:add(nn.ReLU())
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(256, 32))
   model:add(nn.ReLU())
   model:add(nn.Linear(32, noutputs))

elseif opt.model == 'S_p64c5x41_p128c6x5P_p256c5x5P_256f_32f' then
   -- a typical modern convolution network (conv+relu+pool)
   model = nn.Sequential()

   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   local width = opt.kL+opt.kR+1
   model:add(nn.SpatialZeroPadding(0,0,2,2)) -- 21x84 -> 25x84
   model:add(nn.SpatialConvolutionMM(1, 64, 41, 5)) -- 25x84 -> 21x44
   model:add(nn.ReLU())
   --model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 96->48

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 21x44 -> 25x48
   model:add(nn.SpatialConvolutionMM(64, 128, 5, 6)) -- 25x48 -> 20x44
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 20x44 -> 10x22
   width=math.floor((width-1)/poolsize)

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 10x22 -> 14x26
   model:add(nn.SpatialConvolutionMM(128, 256, 5, 5)) -- 14x26 -> 10x22
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 10x22 -> 5x11
   width=math.floor((width)/poolsize)

   -- stage 3 : standard 2-layer neural network
   fully_connected_size = 256*width*11
   model:add(nn.View(fully_connected_size))
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(fully_connected_size, 256))
   model:add(nn.ReLU())
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(256, 32))
   model:add(nn.ReLU())
   model:add(nn.Linear(32, noutputs))

elseif opt.model == 'cuda10_pad' then
   -- a typical modern convolution network (conv+relu+pool)
   model = nn.Sequential()


   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 96->100
   model:add(nn.SpatialConvolutionMM(nfeats, 128, filtsize, filtsize)) -- 100->96
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 96->48


   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 48->52
   model:add(nn.SpatialConvolutionMM(128, 256, filtsize, filtsize)) -- 52->48
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 48->24

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 24->28
   model:add(nn.SpatialConvolutionMM(256, 256, 5, 5)) -- 28->24
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 24->12

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 12->16
   model:add(nn.SpatialConvolutionMM(256, 512, 5, 5)) -- 16->12
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 12->6

   -- stage 3 : standard 2-layer neural network
   fully_connected_size = 512*6*6
   model:add(nn.View(fully_connected_size))
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(fully_connected_size, 512))
   model:add(nn.ReLU())
   model:add(nn.Linear(512, noutputs))





elseif opt.model == 'cuda11_pad' then
   -- a typical modern convolution network (conv+relu+pool)
   model = nn.Sequential()


   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 96->100
   model:add(nn.SpatialConvolutionMM(nfeats, 128, filtsize, filtsize)) -- 100->96
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 96->48


   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 48->52
   model:add(nn.SpatialConvolutionMM(128, 256, filtsize, filtsize)) -- 52->48
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 48->24

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 24->28
   model:add(nn.SpatialConvolutionMM(256, 256, 5, 5)) -- 28->24
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 24->12

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialZeroPadding(2,2,2,2)) -- 12->16
   model:add(nn.SpatialConvolutionMM(256, 512, 5, 5)) -- 16->12
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) -- 12->6

   -- stage 3 : standard 2-layer neural network
   fully_connected_size = 512*6*6
   model:add(nn.View(fully_connected_size))
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(fully_connected_size, 1024))
   model:add(nn.ReLU())
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.Linear(1024, 512))
   model:add(nn.ReLU())
   model:add(nn.Linear(512, noutputs))

else

   error('unknown -model')

end


----------------------------------------------------------------------
---- Visualization is quite easy, using gfx.image().
--
--if opt.visualize then
--   if opt.model == 'convnet' then
--      print '==> visualizing ConvNet filters'
--      gfx.image(model:get(1).weight, {zoom=2, legend='L1'})
--      gfx.image(model:get(5).weight, {zoom=2, legend='L2'})
--   end
--end
