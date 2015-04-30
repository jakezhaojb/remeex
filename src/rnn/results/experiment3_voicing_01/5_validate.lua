require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining validation procedure'

-- track best model
bestaccuracy = bestaccuracy or  0

-- validate function
function validate()
   confusion:zero()
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

   -- validate over validate data
   print('------------------------------VALIDATING--------------------------------------')
   print('                                                 ' .. opt.tag)
   
   --mybatchsize = 100
   print('Batchsize = ' .. batchSize)
   loss = 0
   for s = 1,VALIDATION.songcount do
      xlua.progress(s, VALIDATION.songcount)
      local song = VALIDATION.songmap[s]
      local cqt = VALIDATION.CQT[song]
      local melody = VALIDATION.MELODY[song]

      reset_s()
      segment_length = math.floor(cqt:size(2)/(opt.kL))
      for t = 1,segment_length do
         local from = (t-1)*(opt.kL)+1
         local to = t*(opt.kL)
         local inputs = cqt[{{},{from,to},{}}]
         local targets = melody[{{},{from,to}}]

         if opt.type == 'cuda' then 
           inputs = inputs:cuda() 
           targets = targets:cuda()
         end
         inputs = inputs:cuda()
         targets = targets:cuda()
         loss = forwardpass(inputs,targets) + loss
      end
   end
   print("loss =",loss/VALIDATION.songcount)

   -- timing
   time = sys.clock() - time
   time = time / VALIDATION.songcount
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
