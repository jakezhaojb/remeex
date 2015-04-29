require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining validation procedure'

-- track best model
bestaccuracy = bestaccuracy or  0

-- validate function
function validate()
   collectgarbage()

   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and validating, like Dropout)
   model:evaluate()

   -- create tensor to store indicator for correct/incorrect predictions
   correct = torch.Tensor(validationData:size()):zero() 

   -- validate over validate data
   print('==> validating on validate set:')
   
   batchsize = 32
   print('Batchsize = ' .. batchsize)
   for t = 1,validationData:size(),batchsize do
      -- disp progress
      xlua.progress(t, validationData:size())

      local input = validationData.data[{{t,math.min(t+batchsize-1,validationData:size())}}]
      local target = validationData.labels[{{t,math.min(t+batchsize-1,validationData:size())}}]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end

      local pred = model:forward(input)
      confusion:batchAdd(pred,target)

   end

   -- timing
   time = sys.clock() - time
   time = time / validationData:size()
   print("\n==> time to validate 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- compare to best model
   print('==> best accuracy = ' .. bestaccuracy*100)
   if bestaccuracy < confusion.totalValid then
      -- save/log current net
      local filename = paths.concat(results_folder, 'model_best.net')
      local optimstatepath = paths.concat(results_folder, 'optimState_best.net')
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('==> found better model')
      print('==> saving model to '..filename)
      torch.save(filename, model)
      torch.save(optimstatepath, optimState)
      bestaccuracy = confusion.totalValid
   end

   -- update log/plot
   validationLogger:add{['% mean class accuracy (validation set)'] = confusion.totalValid * 100}

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end
