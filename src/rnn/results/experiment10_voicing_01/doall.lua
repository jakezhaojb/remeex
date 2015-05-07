tag = opt.tag
folder_tag = tag
results_folder = paths.concat(opt.save,folder_tag)
sys.execute('mkdir -p ' .. results_folder)

print(opt)

function doall()
	print('==> run doall')
	-- save out params
	params = io.open(paths.concat(results_folder,'params.txt'),'a')
	params:write('parameter, value','\n')
	for key,val in pairs(opt) do
		params:write(key,' = ',tostring(val),'\n')
	end
	params:flush()
	params:close()


	-- nb of threads and fixed seed (for repeatable experiments)
	if opt.type == 'float' then
	   print('==> switching to floats')
	   torch.setdefaulttensortype('torch.FloatTensor')
	elseif opt.type == 'cuda' then
	   print('==> switching to CUDA')
	   require 'cunn'
	   --cutorch.setDevice(opt.device)
	   torch.setdefaulttensortype('torch.FloatTensor')
	end
	torch.setnumthreads(opt.threads)
	torch.manualSeed(opt.seed)

	----------------------------------------------------------------------
	print '==> executing all'
	local timefile = io.open(paths.concat(results_folder,'times.log'),'a')
	local time = sys.clock()

	bestaccuracy = 0
	trainLogger = nil
	validationLogger = nil
	testLogger = nil


	--if not data_load_complete then dofile (paths.concat(opt.path, '1_data.lua')) end
	print("load data from " .. opt.loadDataFile)
	dofile (paths.concat(opt.path, opt.loadDataFile))
	--print("using model " .. opt.model)
	--dofile (paths.concat(opt.path, '2_model.lua'))
	--dofile (paths.concat(opt.path, '3_loss.lua'))

	print '==> training!'
	local trainFile = opt.trainFile or '4_train.lua'
	for key,val in pairs(opt.trainParams) do
		--batchSize = val.batchSize or 1
		numEpochs = val.numEpochs or opt.maxEpochs
		learningRate = val.learningRate or opt.learningRate
		learningRateDecay = val.learningRateDecay or opt.learningRateDecay
		momentum = val.momentum or opt.momentum
		optimization = val.optimization or opt.optimization

		dofile (paths.concat(opt.path, '2_model.lua'))
		dofile (paths.concat(opt.path, trainFile))
		dofile (paths.concat(opt.path, '5_validate.lua'))
		dofile (paths.concat(opt.path, '6_test.lua'))

		p = 1
		while p <= numEpochs do
			collectgarbage()
			train()
			if opt.runValidation then
				validate()
			end
			if opt.runTest then
				test()
			end
			timefile:write(sys.clock()-time,'\n')
			timefile:flush()
			p = p+1
		end
	end	
	
	timefile:close()
	
	model = nil
	trainLogger = nil
	validationLogger = nil
	testLogger = nil
	TRAIN = nil
	VALIDATION = nil
	TEST = nil
end

doall()
--[[
status, errmsg = xpcall(doall, debug.traceback)

print(errmsg)

pcall(function() 
	runlog = io.open(paths.concat(results_folder,'run.log'),'a')
	runlog:write(tostring(status),'\n')
	runlog:write(tostring(errmsg),'\n')
	runlog:flush()
	runlog:close()
	end)

]]