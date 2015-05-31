require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'cunn'
require 'nngraph'
package.path = package.path .. ';/home/justin/torch_utilities/?.lua'
require 'sopt'
time = sys.clock()

results_folder = 'results/test_LSTM_melody_nllweights_025_test'

modelfile = paths.concat(results_folder,'model_best.net')
model = torch.load(modelfile)

predictionDIR = paths.concat(results_folder,'predictions')
os.execute('mkdir -p ' .. predictionDIR)

opt = model.params
batchSize = 1


dofile '1_data_testing.lua'


model.s = model.start_s
for d = 1, #model.s do
   model.s[d] = model.s[d][{{1,batchSize},{}}]
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
function reset_s()
  for d = 1, #model.s do
      model.s[d]:zero()
  end
end
function forwardpass(inputs,targets)
   local x = inputs
   if model.reshapeInputs then
   x = x:reshape(batchSize,opt.num_inputs,1)
   end
   local y = targets
   local s = model.s
   local s_next = nil
   local pred = nil
   model.err, s_next, pred = unpack(model.core_network:forward({x, y, s}))
   confusion:batchAdd(pred, y)
   g_replace_table(model.s,s_next)
   return model.err, pred
end
function grabSongData(s)
    local song = VALIDATION.songmap[s]
    local cqt = VALIDATION.CQT[song]
    local melody = VALIDATION.MELODY[song]
    return song, cqt, melody
end
function grabMiniBatch(t,cqt,melody)
    local inputs = cqt[{{},t,{}}]:cuda()
    local targets = melody[{{},t}]:cuda()
    return inputs, targets
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


-- classes
classes = {}
for i=1,opt.num_outputs do
  table.insert(classes,tostring(i-1))
end

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

g_disable_dropout(model.rnnL)


-- validate over validate data
print('------------------------------VALIDATING--------------------------------------')
print('                                                 ' .. opt.tag)

--mybatchsize = 100
print('Batchsize = ' .. batchSize)
local numsongs = math.min(VALIDATION.songcount,opt.maxNumSongsVALIDATION)
loss = 0
for s = 1,numsongs do
   xlua.progress(s, numsongs)
   local song, cqt, melody = grabSongData(s)
   local file = io.open(paths.concat(predictionDIR,song .. '.csv'),'w+')
   reset_s()
   for t = 1,melody:size(2) do
      local inputs, targets = grabMiniBatch(t,cqt,melody)
      local loss, pred = forwardpass(inputs,targets)
      local maxval,pred_idx = pred:max(2)
      file:write(pred_idx[1][1],"\n")
   end
   file:flush()
   file:close()
end

-- timing
time = sys.clock() - time
print("\n==> time to predict results = " .. (time*1000) .. 'ms')

-- print confusion matrix
confusion:updateValids()
torch.save(paths.concat(results_folder,'confusion.t7b'),confusion)

confusionfile = io.open(paths.concat(results_folder,'confusion.csv'),'w+')
for i = 1, confusion.mat:size(1) do
  row = {}
  for j = 1, confusion.mat:size(2) do
    table.insert(row,confusion.mat[{i,j}])
  end
  confusionfile:write(table.concat(row,','),'\n')
end
confusionfile:flush()
confusionfile:close()

if opt.num_outputs <= 10 then
  print(confusion)
else
  print(confusion.valids:reshape(1,opt.num_outputs)*100)
end
print('% mean class accuracy (VALIDATION)', confusion.totalValid * 100)


outputfile = io.open(paths.concat(results_folder,'evaluation_best.txt'),'w+')
local VxR, VxF, RPA = evaluateConfusion(confusion)
print('VxR =',VxR*100)
print('VxF =',VxF*100)
print('RPA =',RPA*100)
outputfile:write('VxR =',VxR*100,'\n')
outputfile:write('VxF =',VxF*100,'\n')
outputfile:write('RPA =',RPA*100,'\n')
outputfile:write('% mean class accuracy (validation set)',confusion.totalValid * 100,'\n')
outputfile:flush()
outputfile:close()

