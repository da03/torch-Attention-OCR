require 'image'
require 'paths'
require 'utils'
require 'class'

local DataGen = torch.class('DataGen')

function DataGen:__init(data_root, annotation_fn, max_decoder_l)
    self.imgH = 32
    self.data_root = data_root
    self.annotation_fn = annotation_fn
    self.max_decoder_l = max_decoder_l
    local file, err = io.open(self.annotation_fn, "r")
    if err then 
        file, err = io.open(paths.concat(self.data_root, self.annotation_fn), "r")
        if err then
            print(string.format('Error: Data file %s not found ', self.annotation_fn))
            return
        end
    end
    self.lines = {}
    for line in file:lines() do
        local filename, label = unpack(split(line))
        table.insert(self.lines, {filename, label})
    end
    self.cursor = 1
    self.buffer = {}
end

function DataGen:shuffle()
    shuffle(self.lines)
end

function DataGen:size()
    return #self.lines
end

function DataGen:nextBatch(batch_size)
    while true do
        if self.cursor > #self.lines then
            break
        end
        local img = image.load(paths.concat(self.data_root, self.lines[self.cursor][1]))
        local label_str = self.lines[self.cursor][2]
        local label_list = {1}
        for c in label_str:gmatch"." do
            local l = string.byte(c)
            local vocab_id
            if l > 96 then
                vocab_id = l - 97 + 12
            else
                vocab_id = l - 48 + 13
            end
            table.insert(label_list, vocab_id)
        end
        table.insert(label_list, 2)
        self.cursor = self.cursor + 1
        img = 255.0*image.rgb2y(img)
        local origH = img:size()[2]
        local origW = img:size()[3]
        local imgW = math.ceil(origW *self.imgH / origH)
        img = image.scale(img, imgW, self.imgH)
        if self.buffer[imgW] == nil then
            self.buffer[imgW] = {}
        end
        table.insert(self.buffer[imgW], {img:clone(), label_list})
        if #self.buffer[imgW] == batch_size then
            local images = torch.Tensor(batch_size, 1, self.imgH, imgW)
            local max_target_length = #self.buffer[imgW][1][2]
            for i = 1, #self.buffer[imgW] do
                images[i]:copy(self.buffer[imgW][i][1])
                max_target_length = math.max(max_target_length, #self.buffer[imgW][i][2])
            end
            local targets = torch.IntTensor(batch_size, max_target_length):fill(0)
            for i = 1, #self.buffer[imgW] do
                 for j = 1, #self.buffer[imgW][i][2] do
                     targets[i][j] = self.buffer[imgW][i][2][j] 
                 end
            end
            self.buffer[imgW] = nil
            --collectgarbage()
            do return {images, targets} end
        end
    end

    if next(self.buffer) == nil then
        self.cursor = 0
        return nil
    end
    local imgW, v = next(self.buffer, nil)
    real_batch_size = #self.buffer[imgW]
    local images = torch.Tensor(real_batch_size, 1, self.imgH, imgW)
    local max_target_length = #self.buffer[imgW][1][2]
    for i = 1, #self.buffer[imgW] do
        images[i]:copy(self.buffer[imgW][i][1])
        max_target_length = math.max(max_target_length, #self.buffer[imgW][i][2])
    end
    local targets = torch.IntTensor(batch_size, max_target_length):fill(0)
    for i = 1, #self.buffer[imgW] do
        for j = 1, #self.buffer[imgW][i][2] do
            targets[i][j] = self.buffer[imgW][i][2][j] 
        end
    end
    self.buffer[imgW] = nil
    collectgarbage()
    return {images, targets}
end
