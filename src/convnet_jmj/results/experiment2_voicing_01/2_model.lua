
require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers


----------------------------------------------------------------------
print '==> define parameters'

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

if opt.model == 'linear' then

   -- Simple linear model
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,noutputs))

elseif opt.model == 'mlp' then

   -- Simple 2-layer neural network, with tanh hidden units
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,nhiddens))
   model:add(nn.Tanh())
   model:add(nn.Linear(nhiddens,noutputs))


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
print '==> here is the model:'
print(model)

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
