--[[
Script that visualizes the feature planes or filters
By Jake
--]]


require 'nn'
require 'image'


function output_feature_plane(mod)
   -- Input is a module
   local osize = mod.output:size()
   local Irec = image.toDisplayTensor({input=mod.output:reshape(osize[1]*osize[2],
                                                   osize[3], osize[4]), padding=1}) -- TODO suspect
   return Irec
end


function output_filters(mod)
   Irec = image.toDisplayTensor({input=mod.weight, padding=1})
end


function main()
   a = nn.Sequential()
   a:add(nn.SpatialConvolution(3,8,2,2))
   a:forward(torch.rand(8,3,4,4))
   Irec1 = output_feature_plane(a:get(1))
   Irec2 = output_feature_plane(a:get(1))
   image.save('./1.png', Irec1)
   image.save('./2.png', Irec2)
   print('Done')
end

main()
