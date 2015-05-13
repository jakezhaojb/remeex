require 'torch'   -- torch
require 'nn'
----------------------------------------------------------------------
print '==> loading dataset'

if opt == nil then
	opt = {}
	opt.kL = 10
	opt.kR = 10
	opt.binarylabel = true
	opt.preprocessedData = '/home/justin/cqt/t7b/'
	opt.runValidation = true
	opt.maxNumSamples = 1e9
end


function processdata(CQT,kL,kR,binarylabel,MELODY)
	local IDX = {}
	local n = 0
	for song,cqt in pairs(CQT) do
		if cqt:size()[1] == 84 then
			CQT[song] = CQT[song]:transpose(1,2)
		end
		CQT[song] = CQT[song]:float()
		local cqt = CQT[song]
		IDX[song] = {}
		IDX[song].start = n + 1
		n = n + cqt:size()[1] - opt.kL - opt.kR
		IDX[song].finish = n
		if binarylabel == true then
			MELODY[song] = torch.gt(MELODY[song],0):reshape(cqt:size()[1]):double()+1
		end
	end
	return IDX,n
end

function whichsong(i,IDX)
	-- determines which song index i belongs to
	-- returns nil if no song found
	for song,idx in pairs(IDX) do
		if i >= idx.start and i <= idx.finish then
			return song, i-idx.start+1
		end
	end
	return nil, nil
end

function grabsample(i,DATA)
	local song,j = whichsong(i,DATA.IDX)
	local cqt = DATA.CQT[song][{{j,opt.kL+j+opt.kR},{}}]
	local melody = DATA.MELODY[song][{{j+opt.kL}}]
	return cqt,melody
end


function constructdata(opt,splitname)
	local CQT = torch.load(paths.concat(opt.preprocessedData,splitname .. '_cqt.t7b'))
	local MELODY = torch.load(paths.concat(opt.preprocessedData,splitname .. '_melody.t7b'))
	local IDX, n = processdata(CQT,opt.kL,opt.kR,opt.binarylabel,MELODY)
	DATA = {
		IDX=IDX,
		CQT=CQT,
		MELODY=MELODY,
		kL=opt.kL,
		kR=opt.kR,
		size=function() return n end
	}
	return DATA
end

TRAIN = constructdata(opt,'train')

if opt.runValidation then
	VALIDATION = constructdata(opt,'validation')
	local shuffle = torch.randperm(VALIDATION:size())
	local inputs = {}
	local targets = {}
	for j = 1,math.min(VALIDATION:size(),opt.maxNumSamples) do
		i = shuffle[j]
		local input, target = grabsample(i,VALIDATION)
		table.insert(inputs, input:reshape(1,1,input:size()[1],input:size()[2]))
		table.insert(targets, target)
	end
	-- convert tables back to tensors
	inputs = nn.JoinTable(1):forward(inputs)
	targets = nn.JoinTable(1):forward(targets)
	validationData = {
		data=inputs,
		labels=targets,
		size=function() return inputs:size()[1] end
	}
end

if opt.runTest then
	TEST = constructdata(opt,'test')
end