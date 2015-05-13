--[[
Data class file.
By Jake
--]]


require 'cunn'

local Model = torch.class('Model')

function Model:__init(config)
   self.sequential = Model:createSequential(config)
   self.l1w = config.l1w --TODO
   self.tensortype = torch.getdefaulttensortype()
end


function Model:createSequential(model)
   local new = nn.Sequential()
   for i, m in ipairs(model) do
      new:add(Model:createModule(m))
   end
end

function Model:createModule(m)
   if m.module == 'nn.Linear' then
      return Model:CreateLinear(m)
   elseif m.module == 'nn.ReLU' then
      return Model:CreateReLU(m)
   elseif m.module == 'nn.L1Penalty' then
      return Model:CreateL1Penalty(m)
   else
      error("Unrecognized module for creation: "..tostring(m.module))
   end
end 

function Model:createLinear(m)
   return nn.Linear(m.inputSize, m.outputSize)
end

function Model:createReLU(m)
   return nn.ReLU()
end

function Model:createL1Penalty(m)
   return nn.L1Penalty(m.l1w)
end
