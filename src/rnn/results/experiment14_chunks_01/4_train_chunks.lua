

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
package.path = package.path .. ';/home/justin/torch_utilities/?.lua'
require 'sopt'

----------------------------------------------------------------------

----------------------------------------------------------------------
print '==> defining some tools'


-- classes
classes = {}
for i=1,opt.num_outputs do
  table.insert(classes,tostring(i-1))
end

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)
confusion_chroma = optim.ConfusionMatrix(13)

-- Log results to files
if not trainLogger then
  print('instantiate new trainLogger')
  trainLogger = optim.Logger(paths.concat(results_folder, 'train.log'))
end

if not validationLogger then
  print('instantiate new validationLogger')
  validationLogger = optim.Logger(paths.concat(results_folder, 'validation.log'))
end

if not testLogger then
  print('instantiate new testLogger')
  testLogger = optim.Logger(paths.concat(results_folder, 'test.log'))
end

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector


----------------------------------------------------------------------
print '==> configuring optimizer'


if optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif optimization == 'LBFGS' then
   optimState = {
      learningRate = learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif optimization == 'SGD' then
   optimState = {
      learningRate = learningRate,
      weightDecay = opt.weightDecay,
      learningRateDecay = learningRateDecay,
      momentum = momentum,
      learningRateDecay = 1e-6
   }
   optimMethod = optim.sgd

elseif optimization == 'ASGD' then
   optimState = {
      eta0 = learningRate,
      lambda = opt.lambda,
      t0 = numsamples * opt.t0
   }
   optimMethod = optim.asgd
else
  local lr, mom
  local init_lr = learningRate
  local init_mom = momentum
  local lr_decay = learningRateDecay
  if lr_decay ~= 0 then
      lr = sopt.gentle_decay(init_lr, lr_decay)
  else
      lr = sopt.constant(init_lr)
  end

  if ramp_up_momentum then
      mom = sopt.sutskever_blend(init_mom)
  else
      mom = sopt.constant(init_mom)
  end

  optimState = {
      learning_rate = lr,
      momentum = mom,
      momentum_type = sopt.nag
  }

  if optimization == 'SGU' then
      optimMethod = sopt.sgu
  elseif optimization == 'ADADELTA' then
      optimMethod = sopt.adadelta
  elseif optimization == 'RMSPROP' then
      optimMethod = sopt.rmsprop
  else
     error('unknown optimization method: ' .. tostring(optimization))
  end
end

if reloadOptimState ~= nil then
  optimState = reloadOptimState
end


function g_replace_table(to, from)
  assert(#to == #from)
  for i = 1, #to do
    to[i]:copy(from[i])
  end
end
function g_disable_dropout(node)
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(g_disable_dropout)
    end
    return
  end
  if string.match(node.__typename, "Dropout") then
    node.train = false
  end
end
function g_enable_dropout(node)
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(g_enable_dropout)
    end
    return
  end
  if string.match(node.__typename, "Dropout") then
    node.train = true
  end
end
function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end
function reset_s()
  for j = 1, #model.s do
    for d = 1, #model.s[j] do
      model.s[j][d]:zero()
    end
  end
end
function getminibatch(myshuffle,from,to)
      -- create mini batch
    local inputs = {}
    local targets = {}
    for j = from,to do
       -- load new sample
       i = lookup[myshuffle[j]]
       local input, target = grabsample(i,TRAIN)
       table.insert(inputs, reshapeInputs(input,opt))
       table.insert(targets, reshapeTargets(target))
    end
    -- convert tables back to tensors
    inputs = nn.JoinTable(1):forward(inputs)
    targets = nn.JoinTable(1):forward(targets)
    return inputs, targets
end
function g_disable_dropout(node)
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(g_disable_dropout)
    end
    return
  end
  if string.match(node.__typename, "Dropout") then
    node.train = false
  end
end
function g_enable_dropout(node)
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(g_enable_dropout)
    end
    return
  end
  if string.match(node.__typename, "Dropout") then
    node.train = true
  end
end
function forwardpass(inputs,targets)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, opt.kL do
    local x = inputs[{{},i}]--:reshape(batchSize,opt.num_inputs,1)
    if model.reshapeInputs then
      x = x:reshape(batchSize,opt.num_inputs,1)
    end
    local y = targets[{{},i}]
    local s = model.s[i - 1]
    local s_next = nil
    local pred = nil
    model.err[i], s_next, pred = unpack(model.rnnL[i]:forward({x, y, s}))
    confusion:batchAdd(pred, y)
    g_replace_table(model.s[i],s_next)
  end
  g_replace_table(model.start_s,model.s[opt.kL])
  return model.err:mean()
end
function backwardpass(inputs,targets)
  reset_ds()
  -- create dummy tensor to pass into the prediction output
  -- dummy tensor must be set to all zeros so that backprop is not corrupted
  local dummy = transfer_data(torch.zeros(batchSize,opt.num_outputs))
  for i = opt.kL, 1, -1 do
    local x = inputs[{{},i}]
    local y = targets[{{},i}]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local tmp = model.rnnL[i]:backward({x, y, s},{derr, model.ds, dummy})[3]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  if opt.avgBPTT == true then
    gradParameters:div(opt.kL)
  end
  model.norm_dw = gradParameters:norm()
  if model.norm_dw > opt.max_grad_norm then
    local shrink_factor = opt.max_grad_norm / model.norm_dw
    gradParameters:mul(shrink_factor)
  end
  return gradParameters
end
function forwardpass_chroma(inputs,targets,targets_c)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, opt.kL do
    local x = inputs[{{},i}]--:reshape(batchSize,opt.num_inputs,1)
    if model.reshapeInputs then
      x = x:reshape(batchSize,opt.num_inputs,1)
    end
    local y = targets[{{},i}]
    local c = targets_c[{{},i}]
    local s = model.s[i - 1]
    local s_next = nil
    local predy = nil
    local predc = nil
    model.err[i], chroma_err, s_next, predy, predc = unpack(model.rnnL[i]:forward({x, y, c, s}))
    confusion:batchAdd(predy, y)
    confusion_chroma:batchAdd(predc,c)
    g_replace_table(model.s[i],s_next)
  end
  g_replace_table(model.start_s,model.s[opt.kL])
  return model.err:mean()
end
function backwardpass_chroma(inputs,targets,targets_c)
  reset_ds()
  -- create dummy tensor to pass into the prediction output
  -- dummy tensor must be set to all zeros so that backprop is not corrupted
  local dummy = transfer_data(torch.zeros(batchSize,opt.num_outputs))
  local dummyc = transfer_data(torch.zeros(batchSize,13))
  for i = opt.kL, 1, -1 do
    local x = inputs[{{},i}]
    local y = targets[{{},i}]
    local c = targets_c[{{},i}]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local output = model.rnnL[i]:backward({x, y, c, s},{derr, derr, model.ds, dummy, dummyc})
    local tmp = output[4]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  if opt.avgBPTT == true then
    gradParameters:div(opt.kL)
  end
  model.norm_dw = gradParameters:norm()
  if model.norm_dw > opt.max_grad_norm then
    local shrink_factor = opt.max_grad_norm / model.norm_dw
    gradParameters:mul(shrink_factor)
  end
  return gradParameters
end
function getTrainData(song)
  local cqt = TRAIN.CQT[song]
  local melody = TRAIN.MELODY[song]
  local chroma = TRAIN.CHROMA[song]
  return cqt, melody, chroma
end
function getTrainDataSlice(from,to,cqt,melody,chroma)
  local inputs = cqt[{{},{from,to},{}}]
  local targets = melody[{{},{from,to}}]
  local targets_c
  if opt.chroma then
    targets_c = chroma[{{},{from,to}}]
    targets_c = targets_c:cuda()
  else
    targets_c = nil
  end
  inputs = inputs:cuda() 
  targets = targets:cuda()
  return inputs, targets, targets_c
end
function evaluateConfusion(confusion)
  confusion:updateValids()
  local n = opt.num_outputs
  local x = confusion.mat
  local VxR = torch.sum(x[{{2,n},{2,n}}])/torch.sum(x[{{2,n},{}}])
  local VxF = torch.sum(x[{1,{2,n}}])/torch.sum(x[{1,{}}])
  local RPA = torch.trace(x[{{2,n},{2,n}}])/torch.sum(x[{{2,n},{}}])
  return VxR, VxF, RPA
end

----------------------------------------------------------------------
print '==> defining training procedure'

--parameters,gradParameters = model.core_network:getParameters()
function train()
   confusion:zero()
   confusion_chroma:zero()
   collectgarbage()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   g_enable_dropout(model.rnnL)

   -- shuffle at each epoch
   local numsongs = math.min(TRAIN.songcount,opt.maxNumSongsTRAIN)
   local songshuffle = torch.randperm(numsongs)
   local numchunks = #TRAIN.CHUNKS
   local chunkhuffle = torch.randperm(numchunks)

   -- do one epoch
   print('------------------------------TRAINING--------------------------------------')
   print('                                                 ' .. opt.tag)
   print('device: ',cutorch.getDevice())
   print('optim state:')
   print(optimState)
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for i = 1,numchunks do
      -- disp progress
      collectgarbage()
      xlua.progress(i, numchunks)

      j = chunkhuffle[i]
      local song, from, to = unpack(TRAIN.CHUNKS[j])
      local cqt, melody, chroma = getTrainData(song)
      reset_s()
      local inputs,targets,targets_c = getTrainDataSlice(from,to,cqt,melody,chroma)

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                    -- get new parameters
                    if x ~= parameters then
                      parameters:copy(x)
                    end

                    -- reset gradients
                    gradParameters:zero()

                    if opt.chroma then
                      f              = forwardpass_chroma(inputs,targets,targets_c)
                      gradParameters = backwardpass_chroma(inputs,targets,targets_c)
                    else
                      f              = forwardpass(inputs,targets)
                      gradParameters = backwardpass(inputs,targets)
                    end
                    
                    return f,gradParameters
                  end

    -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end
   end
   model.optimState = optimState

   -- time taken
   time = sys.clock() - time
   time = time / numsongs
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   if opt.num_outputs <= 10 then
      print(confusion)
   else
      confusion:updateValids()
      print(confusion.valids:reshape(1,opt.num_outputs)*100)
      print('% mean class accuracy (TRAIN)', confusion.totalValid * 100)
      local VxR, VxF, RPA = evaluateConfusion(confusion)
      print('VxR =',VxR*100)
      print('VxF =',VxF*100)
      print('RPA =',RPA*100)
   end
   if opt.chroma then
      print(confusion_chroma)
   end

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}

   -- save/log current net
   local filename = paths.concat(results_folder, 'model.net')
   local optimstatepath = paths.concat(results_folder, 'optimState.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   confusion_chroma:zero()
   epoch = epoch + 1
end
