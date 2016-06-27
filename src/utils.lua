-- http://stackoverflow.com/questions/17119804/lua-array-shuffle-not-working
function swap(array, index1, index2)
    array[index1], array[index2] = array[index2], array[index1]
end

function shuffle(array)
    local counter = #array
    while counter > 1 do
        local index = math.random(counter)
        swap(array, index, counter)
        counter = counter - 1
    end
end

function trim(str)
    local newstr = str:match( "^%s*(.-)%s*$" )
    return newstr
end

function split(str)
    local t = {}
    for v in string.gmatch(str, "[^%s]+") do
        table.insert(t, v)
    end
    return t
end

function reset_state(state, batch_l, t)
    if t == nil then
        local u = {}
        for i = 1, #state do
            state[i]:zero()
            table.insert(u, state[i][{{1, batch_l}}])
        end
        return u
    else
        local u = {[t] = {}}
        for i = 1, #state do
            state[i]:zero()
            table.insert(u[t], state[i][{{1, batch_l}}])
        end
        return u
    end      
end
