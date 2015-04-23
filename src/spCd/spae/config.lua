--[[
Configuration of the sparse autoencoder
By Jake
--]]


require 'nn'
require 'modules'

config = {}

-- Training data
config.train_data = {}
config.train_data.path = paths.concat() -- TODO
config.train_data.length = 

-- Testing data
-- TODO need test data?
config.test_data = {}
config.test_data.path = paths.concat() -- TODO
config.test_data.length = 

-- The model
config.model = {}
config.model[1] = {module='', inputFrameSize=0, outputFrameSize=0}
config.model[2] = {module='', inputFrameSize=0, outputFrameSize=0}
config.model[3] = {module='', inputFrameSize=0, outputFrameSize=0}
config.model[4] = {module='', inputFrameSize=0, outputFrameSize=0}
config.model[5] = {module='', inputFrameSize=0, outputFrameSize=0}
config.model[6] = {module='', inputFrameSize=0, outputFrameSize=0}

config.loss = 

-- The training driver
config.train = {}
config.train.rates = {}
config.train.momentum = 0.9
config.train.decay = 1e-6

-- The testing driver
-- TODO needed?
config.test = {}

-- Main
config.main = {}
config.main.eras = 100
config.main.epoches = 5000
config.main.log = 10
config.main.debug = false

