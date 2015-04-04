--[[
One second data source
By Jake
--]]

require 'datasource'
require 'paths'
local Threads = require 'threads'
local Worker = require 'threads.worker'
require 'easythreads'

local OneSecondDatasource, parent = torch.class('OneSecondDatasource', 'Datasource')

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
      params.datadir = '/scratch/jz1672/remeex/features/t7b_type1_raw/'
   params.traindir = params.datadir .. 'train'
   params.trainTable = params.datadir .. 'trTable.lua'
   params.testdir = params.datadir .. 'test'
   params.testTable = params.datadir .. 'teTable.lua'
   param.numTrain = 948
   param.numTest = 594
   param.lenSample = 22016
   param.lenLabel = 172
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


function OneSecondDatasource:load_idx(traindir, idx)
   assert(paths.dirp(traindir), 'train director not found.')
   require(self.params.trainTable)
   for k, v in pairs(train_table) do
      if idx > v then
         -- Doing nothing
      else
         return paths.concat(self.params.traindir, idx_sub .. 't7b')
      end
   end
   -- If the last
   idx_sub = idx - train_table[#train_table]
   return paths.concat(self.params.traindir, idx_sub .. 't7b')
end


function OneSecondDatasource:nextBatch(batchSize, set, trans_flag)
   assert(set == 'train') -- TODO
   assert(batchSize == self.params.batchSize) --for now, batchSize is fixed
   -- Total number of training samples
   local idx = torch.randperm(self.numTrain)[{1,batchSize}]
   local data = torch.rand(batchSize, self.lenSample)
   local label = torch.rand(batchSize, self.lenLabel)
   for i = 1, idx do
      local data_elem = self.load_idx(self.traindir, idx)
      data[{i, {}}]:copy(data_elem)
      label[{i, {}}]:copy(label_elem)
   end
   return {data, label}
end


function OneSecondDatasource:nextIteratedBatch(batchSize, set, idx, trans_flag)
   assert(set == 'train') -- TODO
   assert(batchSize == self.params.batchSize) --TODO
   local data = torch.rand(batchSize, self.lenSample)
   local label = torch.rand(batchSize, self.lenLabel)
   if idx*batchSize > self.numTrain then
      return nil
   end
   for i = 1, batchSize do
      local data_idx = batchSize*(idx-1) + i
      local data_elem = self.load_idx(self.traindir, data_idx)
      data[{i, {}}]:copy(data_elem)
      label[{i, {}}]:copy(label_elem)
   end
   return {data, label}
end
