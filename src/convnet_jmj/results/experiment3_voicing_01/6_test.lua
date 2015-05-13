require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'


-- test function
function test()
   collectgarbage()

   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- create tensor to store indicator for correct/incorrect predictions
   correct = torch.Tensor(testData:size()):zero() 

   -- test over test data
   print('==> testing on test set:')
   
   batchsize = 32
   print('Batchsize = ' .. batchsize)
   for t = 1,testData:size(),batchsize do
      -- disp progress
      xlua.progress(t, testData:size())

      local input = testData.data[{{t,math.min(t+batchsize-1,testData:size())}}]
      local target = testData.labels[{{t,math.min(t+batchsize-1,testData:size())}}]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end

      local pred = model:forward(input)
      confusion:batchAdd(pred,target)

   end
   
   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end
