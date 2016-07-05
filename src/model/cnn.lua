function createCNNModel(use_cuda)
    local model = nn.Sequential()

    --if use_cuda > 0 then
    --    model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', false, true))
    --end
    -- input shape: (None, 1, 32, None)
    -- CNN part
    model:add(nn.AddConstant(-128.0))
    model:add(nn.MulConstant(1.0 / 128))

    model:add(cudnn.SpatialConvolution(1, 64, 3, 3, 1, 1, 1, 1)) -- (None, 64, 32, None)
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- (None, 64, 16, None)

    model:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)) -- (None, 128, 16, None)
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- (None, 128, 8, None)

    model:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)) -- (None, 256, 8, None)
    model:add(nn.SpatialBatchNormalization(256))
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)) -- (None, 256, 8, None)
    model:add(cudnn.ReLU(true))
    
    model:add(cudnn.SpatialMaxPooling(1, 2, 1, 2, 0, 0)) -- (None, 256, 4, None)

    model:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)) -- (None, 512, 4, None)
    model:add(nn.SpatialBatchNormalization(512))
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)) -- (None, 512, 4, None)
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialMaxPooling(1, 2, 1, 2, 0, 0)) -- (None, 512, 2, None)

    model:add(cudnn.SpatialConvolution(512, 512, 2, 2, 1, 1, 0, 0)) -- (None, 512, 1, None)
    model:add(nn.SpatialBatchNormalization(512))
    model:add(cudnn.ReLU(true))

    model:add(nn.View(512, -1):setNumInputDims(3)) -- (None, 512, None)
    model:add(nn.Transpose({2, 3})) -- (None, None, 512)
    --model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', false, true))
    --model:cuda()
    return model

end
