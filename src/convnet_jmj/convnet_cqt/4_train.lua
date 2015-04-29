

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
package.path = package.path .. ';/home/justin/torch_utilities/?.lua'
require 'sopt'

----------------------------------------------------------------------
-- CUDA?
if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

----------------------------------------------------------------------
print '==> defining some tools'

-- numsamples
numsamples = math.min(TRAIN:size(),opt.maxNumSamples)
lookup = torch.randperm(TRAIN:size())

-- classes
classes = {'1','2'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

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
if model then
   parameters,gradParameters = model:getParameters()
end

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

----------------------------------------------------------------------
print '==> defining training procedure'

function train()
    collectgarbage()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(numsamples)

   -- do one epoch
   print('------------------------------TRAINING--------------------------------------')
   print('                                                 ' .. opt.tag)
   print('optim state:')
   print(optimState)
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1,numsamples,batchSize do
      -- disp progress
      collectgarbage()
      xlua.progress(t, numsamples)

      -- create mini batch
      local inputs = {}
      local targets = {}
      for j = t,math.min(t+batchSize-1,numsamples) do
         -- load new sample
         i = lookup[shuffle[j]]
         local input, target = grabsample(i,TRAIN)
         table.insert(inputs, reshapeInputs(input,opt))
         table.insert(targets, target)
      end
      -- convert tables back to tensors
      inputs = nn.JoinTable(1):forward(inputs)
      targets = nn.JoinTable(1):forward(targets)

      if opt.type == 'cuda' then 
        inputs = inputs:cuda() 
        targets = targets:cuda()
      end

      batchSize = inputs:size()[1]
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                      -- get new parameters
                      if x ~= parameters then
                        parameters:copy(x)
                      end

                      -- reset gradients
                      gradParameters:zero()

                      -- f is the average of all criterions
                      local f = 0

                      -- evaluate function for complete mini batch
                      -- estimate f
                      local output = model:forward(inputs)
                      local err = criterion:forward(output, targets)
                      f = f + err

                      -- estimate df/dW
                      local df_do = criterion:backward(output, targets)
                      model:backward(inputs, df_do)

                      -- update confusion
                      confusion:batchAdd(output, targets)
                       
                       -- normalize gradients and f(X)
                       --gradParameters:div(batchSize)
                       --f = f/batchSize

                      -- return f and df/dX
                      return f,gradParameters
                    end

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end
   end

   -- time taken
   time = sys.clock() - time
   time = time / numsamples
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}

   -- save/log current net
   local filename = paths.concat(results_folder, 'model.net')
   local optimstatepath = paths.concat(results_folder, 'optimState.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)
   torch.save(optimstatepath, optimState)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end
