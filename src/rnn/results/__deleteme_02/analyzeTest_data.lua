----------------------------------------------------------------------

----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator


----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('STL-10 Supervised Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
cmd:option('-data', '../../data/stl10_binary', 'subdirectory containing data files')
cmd:option('-saveData', '../../data/stl10_binary', 'subdirectory containing data files')
   cmd:text()
   opt = cmd:parse(arg or {})
end



num_samples = 8000

----------------------------------------------------------------------
print '==> loading dataset'

data_folder = opt.data

-- Open the files and set little endian encoding
data_fd = torch.DiskFile(paths.concat(data_folder,"test_X.bin"), "r", true)
data_fd:binary():littleEndianEncoding()
label_fd = torch.DiskFile(paths.concat(data_folder,"test_y.bin"), "r", true)
label_fd:binary():littleEndianEncoding()

-- Create and read the data
X = torch.ByteTensor(num_samples, 3, 96, 96)
data_fd:readByte(X:storage())
Y = torch.ByteTensor(num_samples)
label_fd:readByte(Y:storage())

-- Because data is in column-major, transposing the last 2 dimensions gives result that can be correctly visualized
X = X:transpose(3, 4)

tesize = num_samples

-- Initialize table testData
testData = {
   data = X,
   labels = Y,
   size = function() return tesize end
}



----------------------------------------------------------------------
print '==> preprocessing data'

-- convert to cuda, for some reason need to set as float first

testData.data = testData.data:float()

channels = {'r','g','b'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.

print '==> preprocessing data: normalize each feature (channel) globally'
mean = {}
std = {}

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = testData.data[{ {},i,{},{} }]:mean()
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   --testData.data[{ {},i,{},{} }]:div(std[i])
end





-- Local normalization
print '==> preprocessing data: normalize all three channels locally'

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(13)

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for c in ipairs(channels) do
   for i = 1,testData:size() do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
   end
end


----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do

   testMean = testData.data[{ {},i }]:mean()
   testStd = testData.data[{ {},i }]:std()

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end

----------------------------------------------------------------------
print '==> save out data'
foldername = opt.saveData
sys.execute('mkdir -p ' .. foldername)


testData.data = testData.data:reshape(testData:size(),3*96*96)

torch.save(paths.concat(foldername,"testData8000_normalized.bin"),testData)
