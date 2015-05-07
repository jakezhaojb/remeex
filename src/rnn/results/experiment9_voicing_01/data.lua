--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

--[[
--   Modifications by Justin Mao-Jones on 4/27/15
--   for A4 - DSGA1008 Deep Learning Assignment
]]


local stringx = require('pl.stringx')
local file = require('pl.file')

local ptb_path = "/home/justin/lstm/data/"

local datatype = 'word'
local function set_datatype(newtype)
  datatype = newtype
end

-- allow for easier switching between word & char data sets
local datasets = function()
  if datatype == 'word' then
    return {
      trainfn = ptb_path .. "ptb.train.txt",
      testfn  = ptb_path .. "ptb.test.txt",
      validfn = ptb_path .. "ptb.valid.txt"
    }
  elseif datatype == 'char' then
    return {
      trainfn = ptb_path .. "ptb.char.train.txt",
      validfn = ptb_path .. "ptb.char.valid.txt"
    }
  else
    error('unknown datatype')
  end
end

local vocab_idx = 0
local vocab_map = {}
local vocab_map_inverse = {} -- inverse of vocab_map

-- Stacks replicated, shifted versions of x_inp
-- into a single matrix of size x_inp:size(1) x batch_size.
local function replicate(x_inp, batch_size)
   local s = x_inp:size(1)
   local x = torch.zeros(torch.floor(s / batch_size), batch_size)
   for i = 1, batch_size do
     local start = torch.round((i - 1) * s / batch_size) + 1
     local finish = start + x:size(1) - 1
     x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
   end
   return x
end

-- map_data will be used in query_sentences to map user input through vocab_map
local function map_data(data)
   data = stringx.split(data)
   --print(string.format("Loading %s, size of data = %d", fname, #data))
   local x = torch.zeros(#data)
   for i = 1, #data do
      if vocab_map[data[i]] == nil then
         vocab_idx = vocab_idx + 1
         vocab_map[data[i]] = vocab_idx
         vocab_map_inverse[vocab_idx] = data[i]
      end
      x[i] = vocab_map[data[i]]
   end
   return x
end

local function load_data(fname)
   local data = file.read(fname)
   data = stringx.replace(data, '\n', '<eos>')
   return map_data(data)
end

local function traindataset(batch_size, char)
   local x = load_data(datasets().trainfn)
   x = replicate(x, batch_size)
   return x
end

-- Intentionally we repeat dimensions without offseting.
-- Pass over this batch corresponds to the fully sequential processing.
local function testdataset(batch_size)
   if testfn then
      local x = load_data(datasets().testfn)
      x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
      return x
   end
end

local function validdataset(batch_size)
   local x = load_data(datasets().validfn)
   x = replicate(x, batch_size)
   return x
end

return {traindataset=traindataset,
        testdataset=testdataset,
        validdataset=validdataset,
        vocab_map=vocab_map,
        vocab_map_inverse=vocab_map_inverse,
        map_data=map_data, -- map_data will be used in query_sentences to map user input through vocab_map
        set_datatype=set_datatype} -- switch between word & char datasets