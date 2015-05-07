function tocsv(filename,x,headers)
	file = io.open(filename,'w+')

	if x:nDimension() == 2 then
		m = x:size()[1]
		n = x:size()[2]
	elseif x:nDimension() == 1 then
		m = x:size()[1]
		n = 1
	else
		error("cannot process tensors with more than 2 dimensions")
	end

	if headers ~= nil then
		for i,header in pairs(headers) do
			file:write(header)
			if i < n then
				file:write(',')
			end
		end
		file:write('\n')
	end

	for i=1,m do
		if n == 1 then
			file:write(x[i],'\n')
		else
			for j=1,n-1 do
				file:write(x[{i,j}],',')
			end
			file:write(x[{i,n}],'\n')
		end
	end
	file:flush()
	file:close()
end