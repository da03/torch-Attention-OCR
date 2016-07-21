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

-- https://gist.github.com/Badgerati/3261142
-- Returns the Levenshtein distance between the two given strings
function string.levenshtein(str1, str2)
	local len1 = string.len(str1)
	local len2 = string.len(str2)
	local matrix = {}
	local cost = 0
	
        -- quick cut-offs to save time
	if (len1 == 0) then
		return len2
	elseif (len2 == 0) then
		return len1
	elseif (str1 == str2) then
		return 0
	end
	
        -- initialise the base matrix values
	for i = 0, len1, 1 do
		matrix[i] = {}
		matrix[i][0] = i
	end
	for j = 0, len2, 1 do
		matrix[0][j] = j
	end
	
        -- actual Levenshtein algorithm
	for i = 1, len1, 1 do
		for j = 1, len2, 1 do
			if (str1:byte(i) == str2:byte(j)) then
				cost = 0
			else
				cost = 1
			end
			
			matrix[i][j] = math.min(matrix[i-1][j] + 1, matrix[i][j-1] + 1, matrix[i-1][j-1] + cost)
		end
	end
	
        -- return the last value - this is the Levenshtein distance
	return matrix[len1][len2]
end

function localize(thing)
    assert (use_cuda ~= nil, 'use_cuda must be set!')
    if use_cuda then
        return thing:cuda()
    end
    return thing
end

function str2numlist(label_str)
    local label_list = {2}
    for c in label_str:gmatch"." do
        local l = string.byte(c)
        local vocab_id
        if l > 96 then -- a: 97, to 13; z: 122, to 38
            vocab_id = l - 97 + 12 + 1
        else -- 0: 48, to 4; 8: 56, to 12
            vocab_id = l - 48 + 3 + 1
            if vocab_id == 13 then
                vocab_id = 39 --9: 57, to 39
            end
        end
        table.insert(label_list, vocab_id)
    end
    table.insert(label_list, 3)
    return label_list
end

function numlist2str(label_list)
    local str = {}
    for i = 1, #label_list do
        local vocab_id = label_list[i]
        local l
        if vocab_id > 12 then
            l = vocab_id - 1 - 12 + 97
            if vocab_id == 39 then
                l = 57 --9
            end
        else
            l = vocab_id - 1 - 3 + 48
        end
        table.insert(str, l)
    end
    label_str = string.char(unpack(str))
    return label_str 
end

function evalWordErrRate(labels, target_labels, visualize)
    local batch_size = labels:size()[1]
    local target_l = labels:size()[2]
    assert(batch_size == target_labels:size()[1])
    assert(target_l == target_labels:size()[2])

    local word_error_rate = 0.0
    local labels_pred = {}
    local labels_gold = {}
    for b = 1, batch_size do
        local label_list = {}
        for t = 1, target_l do
            local label = labels[b][t]
            if label == 3 then
                break
            end
            table.insert(label_list, label)
        end
        local target_label_list = {}
        for t = 1, target_l do
            local label = target_labels[b][t]
            if label == 3 then
                break
            end
            table.insert(target_label_list, label)
        end
        local label_str = numlist2str(label_list)
        local target_label_str = numlist2str(target_label_list)
        if visualize then
            table.insert(labels_pred, label_str)
            table.insert(labels_gold, target_label_str)
        end
        local edit_distance = string.levenshtein(label_str, target_label_str)
        if edit_distance ~= 0 then
            word_error_rate = word_error_rate + 1
        end
        --word_error_rate = word_error_rate + math.min(1,edit_distance / string.len(target_label_str))
    end
    return word_error_rate, labels_pred, labels_gold
end

