function Endswith(String, End)
   return End == '' or string.sub(String, -string.len(End)) == End
end

function Beginswith(String, Begin)
   return Begin == '' or string.sub(String, 1, string.len(Begin)) == Begin
end
