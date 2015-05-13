--[[
Data file
By Jake
--]]

-- TODO
-- Write a thread pool...
-- each thread read from a file and choose some vector meanwhile throw something away.
-- Programming challenging....

require 'datasource'
require 'paths'
local Threads = require 'threads'

local Data, parent = torch.class('Data', 'Datasource')

function sort_by_value(tab)
   function compare(a,b)
       return a[2] < b[2]
   end
   table.sort(tab, compare)
   return tab
end

-- params:
--   batchSize
--   length
--   numTrain
--   numTest
--   lenSample=22016
--   lenLabel=172
--   datadir [optional]
function OneSecondDatasource:__init(params)
   parent.__init(self)
   if params.datadir == nil then
      params.datadir = '/scratch/jakez/remeex/2015_4_16_third/'
   end
   params.traindir = params.datadir .. 'train'
   params.testdir = params.datadir .. 'test'
   dofile(params.datadir .. 'train_table.lua')
   params.numTrain = train_count
   params.train_table = sort_by_value(train_idx_table)
   dofile(params.datadir .. 'test_table.lua')
   params.numTest = test_count
   params.test_table = sort_by_value(test_idx_table)
   params.lenSample = 22016
   params.lenLabel = 172
   self.params = params
   self.tensortype = torch.getdefaulttensortype()
end


function OneSecondDatasource:type(typ)
   self.tensortype = typ
   if typ == 'torch.CudaTensor' then
      self.output_cuda = torch.CudaTensor():resize(self.output:size())
      self.labels_cuda = torch.CudaTensor():resize(self.labels:size())
   end
   return self
end


function OneSecondDatasource:load_idx(set, idx)
   if idx == 0 then
      idx = idx + 1  -- TODO. Throw away one sample when using nextIteratedBatch
   end
   local tb, dir
   if set == 'train' then
      dir = self.params.traindir
      tb = self.params.train_table
   elseif set == 'test' then
      dir = self.params.testdir
      tb = self.params.test_table
   else
      error('This set not supported.')
   end
   assert(paths.dirp(dir), 'directory not found')
   for i = 1, #tb do
      if idx > tb[i][2] then
         -- Doing nothing
      else
         local idx_sub = idx - tb[i-1][2] - 1
         return paths.concat(dir, tb[i-1][1], idx_sub .. '.t7b')
      end
   end
   -- If the last
   local idx_sub = idx - tb[#tb][2]
   return paths.concat(dir, tb[#tb][1], idx_sub .. '.t7b')
end


function OneSecondDatasource:nextBatch(batchSize, set, trans_flag)
   assert(batchSize == self.params.batchSize) --for now, batchSize is fixed
   -- Total number of training samples
   local idx
   if set == 'train' then
      idx = torch.random(self.params.numTrain-batchSize)
   elseif set == 'test' then 
      idx = torch.random(self.params.numTest-batchSize)
   else
      error('This set not supported.')
   end
   local data = torch.rand(batchSize, 1, 1, self.params.lenSample)  -- First 1 is nChannel, and second is flattened factor
   local label = torch.rand(batchSize, self.params.lenLabel)
   for i = 1, batchSize do
      local data_elem_path = self:load_idx(set, idx+i-1) -- TODO
      local data_elem = torch.load(data_elem_path)
      data[{i,{},{},{}}]:copy(data_elem[1])
      label[{i, {}}]:copy(data_elem[2])
   end
   return {data, label}
end


function OneSecondDatasource:nextIteratedBatch(batchSize, set, idx, trans_flag)
   --assert(batchSize == self.params.batchSize) --TODO
   local data = torch.rand(batchSize, 1, 1, self.params.lenSample)  -- First 1 is nChannel, and second is flattened factor
   local label = torch.rand(batchSize, self.params.lenLabel)
   local this_set_num
   if set == 'train' then
      this_set_num = self.params.numTrain
   elseif set == 'test' then
      this_set_num = self.params.numTest
   else
      error('This set not supported.')
   end
   if (idx-1)*batchSize > this_set_num then
      return nil
   else
      local length = math.min(this_set_num, idx*batchSize) - ((idx-1)*batchSize)
      for i = 1, length do
         local data_idx = batchSize*(idx-1) + i - 1
         local data_elem_path = self:load_idx(set, data_idx)
         local data_elem = torch.load(data_elem_path)
         data[{i,{},{},{}}]:copy(data_elem[1])
         label[{i, {}}]:copy(data_elem[2])
      end
   end
   return {data, label}
end
