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

if opt.binarylabel == true then
	opt.num_outputs = 2
else
	opt.num_outputs = 85
end


function processdata(CQT,kL,kR,binarylabel,MELODY)
	local IDX = {}
	local n = 0
	local means = {}
	local stds = {}
	local lengths = {}
	for song,cqt in pairs(CQT) do
		if cqt:size()[1] == 84 then
			CQT[song] = CQT[song]:transpose(1,2)
		end
		CQT[song] = CQT[song]:float()
		local cqt = CQT[song]
		IDX[song] = {}
		IDX[song].start = n + 1
		lengths[song] = cqt:size()[1]
		n = n + lengths[song] - opt.kL - opt.kR
		IDX[song].finish = n
		means[song] = torch.mean(cqt,1)
		stds[song] = torch.std(cqt,1)
		if binarylabel == true then
			MELODY[song] = torch.gt(MELODY[song],0):reshape(cqt:size()[1]):double()+1
		end
	end
	return IDX,n,means,stds,lengths
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
	local melody = DATA.MELODY[song][{{j,opt.kL+j+opt.kR}}]
	return cqt,melody
end


function constructdata(opt,splitname)
	local CQT = torch.load(paths.concat(opt.preprocessedData,splitname .. '_cqt.t7b'))
	local MELODY = torch.load(paths.concat(opt.preprocessedData,splitname .. '_melody.t7b'))
	local IDX,n,means,stds,lens = processdata(CQT,opt.kL,opt.kR,opt.binarylabel,MELODY)
	DATA = {
		IDX=IDX,
		CQT=CQT,
		MELODY=MELODY,
		kL=opt.kL,
		kR=opt.kR,
		means=means,
		stds=stds,
		lens=lens,
		size=function() return n end
	}
	return DATA
end

function get_mean_std(data)
	local means = nn.JoinTable(data.means)
	local lens = torch.expand(nn.JoinTable(data.lens):reshape(means:size()[1]),means:size())
	local stds = nn.JoinTable(data.stds)

	local mean_freq = torch.sum(torch.cmul(means,lens)/data:size(),1)
	local mean_global = mean_freq:mean()
	local std_freq = torch.pow(torch.sum(torch.cmul(torch.pow(stds,2),lens-1)/data:size(),1),0.5)
	local std_global = torch.pow(torch.sum(torch.cmul(torch.pow(stds,2),lens-1)/data:size()),0.5)
	return mean_freq,mean_global,std_freq,std_global
end


TRAIN = constructdata(opt,'train')
--mean_freq,mean_global,std_freq,std_global = get_mean_std(TRAIN)

--[[
function normalize(data,mean_freq,mean_global,std_freq,std_global,opt)
	if opt.normalizeSTD ~= true then
		std_freq,std_global = torch.zeros(1,84),1
	end
	for song,cqt in pairs(data.CQT) do
		if opt.normalizeFREQ then
			data.CQT[song] = torch.cdiv(cqt-torch.expand(mean_freq,cqt:size()),torch.expand(std_freq,cqt:size()))
		else
			data.CQT[song] = (cqt-mean_global)/std_global
		end
	end
	checkmean,checkstd = get_mean_std(data)
	print("check mean: ",torch.max(torch.abs(checkmean)))
	print("check std: ",torch.min(checkstd),torch.max(checkstd))
end
]]

function normalize(data,opt)
	for song,cqt in pairs(data.CQT) do
		local std = 1
		if opt.normalizeSTD then
			std = cqt:std()
		end
		if opt.normalizeFREQ then
			mean = torch.mean(cqt,1)
			data.CQT[song] = (cqt-torch.expand(mean,cqt:size()))/std
		else
			data.CQT[song] = (cqt-cqt:mean())/std
		end
		checkmean = data.CQT[song]:mean()
		checkstd = data.CQT[song]:std()
		print('checkmean =',checkmean,'checkstd =',checkstd,'max(abs(mean))',torch.max(torch.abs(torch.mean(data.CQT[song],1))))
	end
end



if opt.normalize == true then
	print ">> Normalize TRAIN"
	--normalize(TRAIN,mean_freq,mean_global,std_freq,std_global,opt)
	normalize(TRAIN,opt)
end

function reshapeInputs(input,opt)
	if opt.mode == 'Temporal' then
		return input:reshape(1,input:size()[1],input:size()[2])
	elseif opt.mode == 'Spatial' then 
		return input:reshape(1,1,input:size()[1],input:size()[2])
	else
		error('unknown opt.mode = ' .. tostring(opt.mode))
	end
end

function reshapeTargets(target)
	return target:reshape(1,target:size()[1])
end

if opt.runValidation then
	VALIDATION = constructdata(opt,'validation')
	if opt.normalize == true then
		print ">> Normalize VALIDATION"
		normalize(VALIDATION,opt)
	end
	local shuffle = torch.randperm(VALIDATION:size())
	local inputs = {}
	local targets = {}
	for j = 1,math.min(VALIDATION:size(),opt.maxNumSamplesValidation) do
		i = shuffle[j]
		local input, target = grabsample(i,VALIDATION)
		table.insert(inputs, reshapeInputs(input,opt))
		table.insert(targets, reshapeTargets(target))
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




