--[[
Auxiliary code
By Jake
--]]

function normalize_filters(w)
   local wsz = w:size()
   local w2 = w:transpose(1,2):contiguous():resize(wsz[2],wsz[1]*wsz[3]*wsz[4])
   local norm = w2:norm(2,2):expandAs(w2):contiguous()
   w2:cdiv(norm)
   w:copy(w2:resize(wsz[2],wsz[1],wsz[3],wsz[4]):transpose(1,2))
   collectgarbage()
end


local function reverse(x)
   local n = x:size(1)
   for i = 1,math.floor(n/2) do
      local tmp = x[i]
      x[i] = x[n-i+1]
      x[n-i+1] = tmp
   end
   return x
end

function flip_weights(W)
   local Wt = W:clone():transpose(1,2)
   for i = 1,Wt:size(1) do
      for j = 1,Wt:size(2) do
	 for n = 1,Wt:size(3) do
	    reverse(Wt[i][j][n])
	 end
	 for n = 1,Wt:size(4) do
	    reverse(Wt[i][j]:select(2,n))
	 end
      end
   end
   return Wt
end 

function convert_option(s)
   local out = {}
   local args = string.split(s, '-')
   for _, x in pairs(args) do
      local y = tonumber(x)
      if y == nil then
	 error("Parsing arguments: " .. s .. " is not well formed")
      end
      out[1+#out] = y
   end
   return out
end

function isTableEqual(a, b)
   assert(type(a) == 'table' and type(b) == 'table')
   if #a ~= #b then
      return false
   end
   for i = 1, #a do
      if a[i] ~= b[i] then
         return false
      end
   end
   return true
end

function number_true(tb)
   assert(type(tb) == 'table')
   local n = 0
   for i = 1, #tb do
      if tb[i] then
         n = n + 1
      end
   end
   return n
end

function table_to_bool(tb)
   assert(type(tb) == 'table')
   tb_ = {}
   for i = 1, #tb do
      if tb[i] ~= 0 then
         tb_[i] = true
      else
         tb_[i] = false
      end
   end
   return tb_
end

function check_nan_number(n)
   return n ~= n
end

function check_nan_tensor(ts)
   return check_nan_number(ts:sum())
end
