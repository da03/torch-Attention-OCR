tds = require('tds')

if logging ~= nil then
    log = function(msg) logging:info(msg) end
else
    log = print
end
function reduce(list)
    local acc
    for k, v in ipairs(list) do
        if 1 == k then
            acc = v
        else
            acc = acc +  v
        end
    end
    return acc
end
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
        if l > 96 then -- a: 97, to 14; z: 122, to 39
            vocab_id = l - 97 + 13 + 1
        else -- 0: 48, to 4; 9: 57, to 13
            vocab_id = l - 48 + 3 + 1
        end
        table.insert(label_list, vocab_id-3)
    end
    table.insert(label_list, 3)
    return label_list
end

function numlist2str(label_list)
    local str = {}
    for i = 1, #label_list do
        local vocab_id = label_list[i] + 3
        local l
        if vocab_id > 13 then
            l = vocab_id - 1 - 13 + 97
        else
            l = vocab_id - 1 - 3 + 48
        end
        table.insert(str, l)
    end
    label_str = string.char(unpack(str))
    return label_str 
end

function evalWordErrRate(labels, ctc_labels, visualize)
    local batch_size = #labels

    local word_error_rate = 0.0
    local labels_pred = {}
    local labels_gold = {}
    for b = 1, batch_size do
        local label_str = numlist2str(labels[b])
        local target_label_str = numlist2str(ctc_labels[b])
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

function loadDictionary(dictionary_path, allow_digit_prefix)
    local file, err = io.open(dictionary_path, "r")
    if err then
        log(string.format('Error: Data file %s not found ', self.data_path))
        os.exit()
    end
    local trie = tds.Hash()
    trie[2] = tds.Hash() -- start symbol
    local idx = 0
    for line in file:lines() do
        idx = idx + 1
        if idx % 1000000==0 then
            log (string.format('%d lines read', idx))
        end
        local str = trim(line)
        local node = trie[2]
        if allow_digit_prefix then
            node[3] = trie[2] -- allow output nothing
            for l = 48, 57 do
                vocab_id = l - 48 + 3 + 1
                node[vocab_id] = trie[2]
            end
        end
        for c in str:gmatch"." do
            local l = string.byte(c)
            local vocab_id
            if l > 96 then -- a: 97, to 14; z: 122, to 39
                vocab_id = l - 97 + 13 + 1
            else -- 0: 48, to 4; 9: 57, to 13
                vocab_id = l - 48 + 3 + 1
            end
            if node[vocab_id] == nil then
                node[vocab_id] = tds.Hash()
            end
            node = node[vocab_id]
        end
        if node[3] == nil then
            node[3] = tds.Hash()
        end
    end
    return trie
end
