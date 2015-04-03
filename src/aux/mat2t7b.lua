-- This file convert .mat to .t7b

require 'mattorch'
require 'aux'

cmd = torch.CmdLine()
cmd:option('--dst', '/tmp/dst.t7b', 'path/to/dstfile')
cmd:option('--src', '/tmp/src.mat', 'path/to/srcfile')
opt = cmd:parse(arg)

assert(paths.filep(opt.src), 'src file not exists.')
assert(Endswith(opt.src, '.mat'), 'src file is .mat')
assert(Endswith(opt.dst, '.t7b'), 'dst file is .t7b')
-- TODO write an interactive inteface here.
--assert(not paths.filep(opt.dst), 'dst file already exist, overwrite?')

data = mattorch.load(opt.src)
data_copy = {}
data_copy[1] = data['1']
data_copy[2] = data['2']

torch.save(opt.dst, data_copy)
print('Conversion done.')
