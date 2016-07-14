 --[[ Load image data. Adapted from https://github.com/da03/Attention-OCR/blob/master/src/data_util/data_gen.py. 
 --    ARGS:
 --        - `data_base_dir`      : string, The base directory of the image path in data_path. If the image path in data_path is absolute path, set it to /.
 --        - `data_path`  : string, The path containing data file names and labels. Format per line: image_path characters. Note that the image_path is the relative path to data_base_dir
 --        - `max_aspect_ratio`  : float, The maximum allowed aspect ratio of resized image. As we set the maximum number of cloned decoders, we need to make sure that the image features sequence length does not exceed the number of available decoders. Image features sequence length is equal to max( img_width/img_height, max_aspect_ratio) * 32 / 4 - 1.
 --]]
require 'image'
require 'paths'
require 'utils'
require 'class'
tds = require('tds')

local DataGen = torch.class('DataGen')

function DataGen:__init(data_base_dir, data_path, max_aspect_ratio)
    self.imgH = 32
    self.data_base_dir = data_base_dir
    self.data_path = data_path
    self.max_width = max_width
    self.max_aspect_ratio = max_aspect_ratio
    self.min_aspect_ratio = 0.5

    if logging ~= nil then
        log = function(msg) logging:info(msg) end
    else
        log = print
    end
    local file, err = io.open(self.data_path, "r")
    if err then 
        file, err = io.open(paths.concat(self.data_base_dir, self.data_path), "r")
        if err then
            log(string.format('Error: Data file %s not found ', self.data_path))
            os.exit()
            --return
        end
    end
    self.lines = tds.Hash()
    idx = 0
    for line in file:lines() do
        idx = idx + 1
        if idx % 1000000==0 then
            log (string.format('%d lines read', idx))
        end
        local filename, label = unpack(split(line))
        self.lines[idx] = tds.Vec({filename, label})
    end
    collectgarbage()
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
        local status, img = pcall(image.load, paths.concat(self.data_base_dir, self.lines[self.cursor][1]))
        if not status then
            self.cursor = self.cursor + 1
        else
            local label_str = self.lines[self.cursor][2]
            local label_list = str2numlist(label_str)
            self.cursor = self.cursor + 1
            img = 255.0*image.rgb2y(img)
            local origH = img:size()[2]
            local origW = img:size()[3]
            local aspect_ratio = origW / origH
            aspect_ratio = math.min(aspect_ratio, self.max_aspect_ratio)
            aspect_ratio = math.max(aspect_ratio, self.min_aspect_ratio)
            local imgW = math.ceil(aspect_ratio *self.imgH)
            --imgW = 100
            img = image.scale(img, imgW, self.imgH)
            if self.buffer[imgW] == nil then
                self.buffer[imgW] = {}
            end
            table.insert(self.buffer[imgW], {img:clone(), label_list})
            if #self.buffer[imgW] == batch_size then
                local images = torch.Tensor(batch_size, 1, self.imgH, imgW)
                local max_target_length = -math.huge
                for i = 1, #self.buffer[imgW] do
                    images[i]:copy(self.buffer[imgW][i][1])
                    max_target_length = math.max(max_target_length, #self.buffer[imgW][i][2])
                end
                -- targets: use as input. SOS, ch1, ch2, ..., chn
                local targets = torch.IntTensor(batch_size, max_target_length-1):fill(1)
                -- targets_eval: use for evaluation. ch1, ch2, ..., chn, EOS
                local targets_eval = torch.IntTensor(batch_size, max_target_length-1):fill(1)
                local num_nonzeros = 0
                for i = 1, #self.buffer[imgW] do
                    num_nonzeros = num_nonzeros + #self.buffer[imgW][i][2] - 1
                    for j = 1, #self.buffer[imgW][i][2]-1 do
                        targets[i][j] = self.buffer[imgW][i][2][j] 
                        targets_eval[i][j] = self.buffer[imgW][i][2][j+1] 
                    end
                end
                self.buffer[imgW] = nil
                --collectgarbage()
                do return {images, targets, targets_eval, num_nonzeros} end
            end
        end
    end

    if next(self.buffer) == nil then
        self.cursor = 0
        collectgarbage()
        return nil
    end
    local imgW, v = next(self.buffer, nil)
    real_batch_size = #self.buffer[imgW]
    local images = torch.Tensor(real_batch_size, 1, self.imgH, imgW)
    local max_target_length = -math.huge
    for i = 1, #self.buffer[imgW] do
        images[i]:copy(self.buffer[imgW][i][1])
        max_target_length = math.max(max_target_length, #self.buffer[imgW][i][2])
    end
    local targets = torch.IntTensor(real_batch_size, max_target_length-1):fill(0)
    local targets_eval = torch.IntTensor(real_batch_size, max_target_length-1):fill(1)
    local num_nonzeros = 0
    for i = 1, #self.buffer[imgW] do
        num_nonzeros = num_nonzeros + #self.buffer[imgW][i][2] - 1
        for j = 1, #self.buffer[imgW][i][2]-1 do
            targets[i][j] = self.buffer[imgW][i][2][j] 
            targets_eval[i][j] = self.buffer[imgW][i][2][j+1] 
        end
    end
    self.buffer[imgW] = nil
    --collectgarbage()
    return {images, targets, targets_eval, num_nonzeros}
end
