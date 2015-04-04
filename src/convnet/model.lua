--[[
model scripts
By Jake
--]]


require 'cunn'

function get_conv_pool(nInputs, nOutputs, kSize, stride, poolSize)
   local struct = nn.Sequential()
   struct:add(nn.TemporalConvolution(nInputs, nOutputs, kSize, stride))
   struct:add(nn.ReLU())
   struct:add(nn.TemporalMaxPooling(poolSize))
   return struct
end


function get_conv(nInputs, nOutputs, kSize, stride)
   local struct = nn.Sequential()
   struct:add(nn.TemporalConvolution(nInputs, nOutputs, kSize, stride))
   struct:add(nn.ReLU())
   return struct
end


function get_softmax(nClasses, nInputs, inputPlaneSize)
   local softmax = nn.Sequential()
   softmax:add(nn.View(nInputs*inputPlaneSize[1]*inputPlaneSize[2])
	       :setNumInputDims(3))
   softmax:add(nn.Linear(nInputs*inputPlaneSize[1]*inputPlaneSize[2], nClasses))
   softmax:add(nn.LogSoftMax())
   return softmax
end


function get_softmax_dropout(nClasses, nInputs, inputPlaneSize, pDropout, nTransition)
   pDropout = pDropout or 0.5
   nTransition = nTransition or {1024, 512}
   assert(type(pDropout) == 'number')
   assert(type(nTransition) == 'table')
   local softmax = nn.Sequential()
   softmax:add(nn.View(nInputs*inputPlaneSize)
	       :setNumInputDims(3))
   softmax:add(nn.Dropout(pDropout))
   softmax:add(nn.Linear(nInputs*inputPlaneSize, nTransition[1]))
   softmax:add(cudnn.ReLU())
   softmax:add(nn.Dropout(pDropout))
   softmax:add(nn.Linear(nTransition[1], nTransition[2]))
   softmax:add(cudnn.ReLU())
   softmax:add(nn.Linear(nTransition[2], nClasses))
   softmax:add(nn.LogSoftMax())
   return softmax
end
