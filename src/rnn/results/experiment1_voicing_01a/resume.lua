require 'optim'


tag = opt.tag
folder_tag = tag
results_folder = paths.concat(opt.save,folder_tag)
sys.execute('mkdir -p ' .. results_folder)

-- need to trim trailing spaces in header of log files
function trim(s)
  -- from PiL2 20.4
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

function reloadLogger(results_folder,logname)
	local Logger = nil
	-- Reload trainlogger
	if not pcall(function()
			local logfile = logname .. ".log"
			local oldlogfile = "old_" .. logname .. ".log"
			sys.execute('cp ' .. paths.concat(results_folder,logfile) .. ' ' .. paths.concat(results_folder,oldlogfile))
			Logger = optim.Logger(paths.concat(results_folder, logfile))
			local header = ""
			local first = true
			for line in io.lines(paths.concat(results_folder,oldlogfile)) do
				if first then
					header = trim(line)
					first = false
				else
					x = tonumber(line)
					if type(x) == 'number' then
						Logger:add{[header] = x}
					end
				end
			end
			print("Successfully imported " .. logname .. "Logger")
		end)
	then print("Failed to load " .. logname .. "Logger") end
	return Logger
end

function reloadTimeLogger(results_folder)
	local maxtime = 0
	local timefile = nil
	if pcall(function() timefile = io.open(paths.concat(results_folder,'times.log'),'r') end) then
		if timefile ~= nil then
			for line in timefile:lines() do
				num = pcall(function() return tonumber(line) end)
				if num then
					maxtime = math.max(maxtime,line)
				end
			end
			timefile:close()
		end
	end
	local timefile = io.open(paths.concat(results_folder,'times.log'),'a+')
	local time = sys.clock() - maxtime
	return time, timefile
end

function doall()

	-- nb of threads and fixed seed (for repeatable experiments)
	if opt.type == 'float' then
	   print('==> switching to floats')
	   torch.setdefaulttensortype('torch.FloatTensor')
	elseif opt.type == 'cuda' then
	   print('==> switching to CUDA')
	   require 'cunn'
	   cutorch.setDevice(opt.device)
	   torch.setdefaulttensortype('torch.FloatTensor')
	end
	torch.setnumthreads(opt.threads)
	torch.manualSeed(opt.seed)



	----------------------------------------------------------------------
	print '==> executing all'
	bestaccuracy = 0
	trainLogger = reloadLogger(results_folder,"train")
	validationLogger = reloadLogger(results_folder,"validation")
	testLogger = reloadLogger(results_folder,"test")
	time,timefile = reloadTimeLogger(results_folder)


	dofile (paths.concat(opt.path, opt.loadDataFile))

	-- Try to load model.  If it doesn't work, create a new model.
	if not pcall(function()	
							model = torch.load(paths.concat(results_folder,'model_best.net')) 
							parameters, gradParameters = model.core_network:getParameters()

					end) then
		dofile (paths.concat(opt.path, '2_model.lua'))
	end

	--if not pcall(function()	reloadOptimState = torch.load(paths.concat(results_folder,'optimState_best.net')) end) then
	--	print(">> could not reload optim state")
	--end

	-- Define loss criterion
	--dofile (paths.concat(opt.path, '3_loss.lua'))
	
	----------------------------------------------------------------------
	print '==> training!'

	local trainFile = opt.trainFile or '4_train.lua'

	for key,val in pairs(opt.trainParams) do
		--batchSize = val.batchSize or 1
		numEpochs = val.numEpochs or opt.maxEpochs
		learningRate = val.learningRate or opt.learningRate
		learningRateDecay = val.learningRateDecay or opt.learningRateDecay
		momentum = val.momentum or opt.momentum
		optimization = val.optimization or opt.optimization

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
	trainData = nil
	validationData = nil
	testData = nil
	TRAIN = nil
	VALIDATION = nil
	TEST = nil
end


status, errmsg = xpcall(doall, debug.traceback)

runlog = io.open(paths.concat(results_folder,'run.log'),'a')
runlog:write(tostring(status),'\n')
runlog:write(tostring(errmsg),'\n')
runlog:flush()
runlog:close()

