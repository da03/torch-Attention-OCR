require 'image'
require 'paths'
require 'utils'
require 'class'

local DataGen = torch.class('DataGen')

function DataGen:__init(data_root, annotation_fn)
    self.imgH = 32
    self.data_root = data_root
    self.annotation_fn = annotation_fn
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
end

function DataGen:shuffle()
    shuffle(self.lines)
end

function DataGen:size()
    return #self.lines
end

function DataGen:nextBatch(batch_size)
    while true do
        img = image.load(paths.concat(self.data_root, self.lines[self.cursor][1]))
        self.cursor = self.cursor + 1
        img = image.rgb2y(img)
        print (img:size())
        img = image.scale(img, imgW, imgH)
        collectgarbage()
    end
end
