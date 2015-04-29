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
   g_disable_dropout(model.rnnL)
   g_disable_dropout(model.rnnt)


   -- create tensor to store indicator for correct/incorrect predictions
   correct = torch.Tensor(validationData:size()):zero() 

   -- validate over validate data
   print('------------------------------VALIDATING--------------------------------------')
   print('                                                 ' .. opt.tag)
   
   mybatchsize = 100
   print('Batchsize = ' .. mybatchsize)
   for t = 1,validationData:size(),mybatchsize do
      -- disp progress
      xlua.progress(t, validationData:size())

      local inputs = validationData.data[{{t,math.min(t+mybatchsize-1,validationData:size())}}]
      local targets = validationData.labels[{{t,math.min(t+mybatchsize-1,validationData:size())}}]

      inputs = inputs:cuda()
      targets = targets:cuda()
      forwardpass(inputs,targets)

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
