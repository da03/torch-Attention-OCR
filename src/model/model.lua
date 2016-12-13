 --[[ Model, adapted from https://github.com/harvardnlp/seq2seq-attn/blob/master/train.lua
--]]
require 'nn'
require 'hdf5'
require 'cudnn'
require 'optim'
require 'paths'

package.path = package.path .. ';src/?.lua' .. ';src/utils/?.lua' .. ';src/model/?.lua' .. ';src/optim/?.lua'
require 'cnn'
require 'LSTM'
require 'output_projector'
require 'criterion'
require 'model_utils'
require 'optim_sgd'
require 'memory'
require 'warp_ctc'

local model = torch.class('Model')

--[[ Args: 
-- config.load_model
-- config.model_dir
-- config.dropout
-- config.encoder_num_hidden
-- config.encoder_num_layers
-- config.max_encoder_l
-- config.batch_size
--]]

-- init
function model:__init()
    if logging ~= nil then
        log = function(msg) logging:info(msg) end
    else
        log = print
    end
end

-- load model from model_path
function model:load(model_path, config)
    config = config or {}

    -- Build model

    assert(paths.filep(model_path), string.format('Model %s does not exist!', model_path))

    local checkpoint = torch.load(model_path)
    local model, model_config = checkpoint[1], checkpoint[2]
    preallocateMemory(config.prealloc)
    self.cnn_model = model[1]:double()
    self.encoder_fw = model[2]:double()
    self.encoder_bw = model[3]:double()
    self.output_projector = model[4]:double()
    self.global_step = checkpoint[3]
    self.optim_state = checkpoint[4]

    -- Load model structure parameters
    self.cnn_feature_size = 512
    self.dropout = model_config.dropout
    self.encoder_num_hidden = model_config.encoder_num_hidden
    self.encoder_num_layers = model_config.encoder_num_layers
    self.target_vocab_size = model_config.target_vocab_size
    self.prealloc = config.prealloc

    self.max_encoder_l = config.max_encoder_l or model_config.max_encoder_l
    self.batch_size = config.batch_size or model_config.batch_size
    self.prealloc = config.prealloc
    self:_build()
end

-- create model with fresh parameters
function model:create(config)
    self.cnn_feature_size = 512
    self.dropout = config.dropout
    self.encoder_num_hidden = config.encoder_num_hidden
    self.encoder_num_layers = config.encoder_num_layers
    self.target_vocab_size = config.target_vocab_size
    self.max_encoder_l = config.max_encoder_l
    self.batch_size = config.batch_size
    self.prealloc = config.prealloc

    preallocateMemory(config.prealloc)

    -- CNN model, input size: (batch_size, 1, 32, width), output size: (batch_size, sequence_length, 512)
    self.cnn_model = createCNNModel()
    self.encoder_fw = createLSTM(self.cnn_feature_size, self.encoder_num_hidden, self.encoder_num_layers, self.dropout, false, false, false, nil, self.batch_size, self.max_encoder_l, 'encoder-fw')
    self.encoder_bw = createLSTM(self.cnn_feature_size, self.encoder_num_hidden, self.encoder_num_layers, self.dropout, false, false, false, nil, self.batch_size, self.max_encoder_l, 'encoder-bw')
    self.output_projector = createOutputUnit(2*self.encoder_num_hidden, self.target_vocab_size)
    self.global_step = 0

    self.optim_state = {}
    self.optim_state.learningRate = config.learning_rate
    self:_build()
end

-- build
function model:_build()
    log(string.format('cnn_featuer_size: %d', self.cnn_feature_size))
    log(string.format('dropout: %f', self.dropout))
    log(string.format('encoder_num_hidden: %d', self.encoder_num_hidden))
    log(string.format('encoder_num_layers: %d', self.encoder_num_layers))
    log(string.format('target_vocab_size: %d', self.target_vocab_size))
    log(string.format('max_encoder_l: %d', self.max_encoder_l))
    log(string.format('batch_size: %d', self.batch_size))
    log(string.format('prealloc: %s', self.prealloc))

    self.config = {}
    self.config.dropout = self.dropout
    self.config.encoder_num_hidden = self.encoder_num_hidden
    self.config.encoder_num_layers = self.encoder_num_layers
    self.config.target_vocab_size = self.target_vocab_size
    self.config.max_encoder_l = self.max_encoder_l
    self.config.batch_size = self.batch_size
    self.config.prealloc = self.prealloc

    if self.optim_state == nil then
        self.optim_state = {}
    end

    -- convert to cuda if use gpu
    self.layers = {self.cnn_model, self.encoder_fw, self.encoder_bw, self.output_projector}
    for i = 1, #self.layers do
        localize(self.layers[i])
    end

    self.context_proto = localize(torch.zeros(self.batch_size, self.max_encoder_l, 2*self.encoder_num_hidden))
    self.ctc_acts_proto = localize(torch.zeros(self.batch_size*self.max_encoder_l, self.target_vocab_size))
    self.ctc_grads_proto = localize(torch.zeros(self.batch_size*self.max_encoder_l, self.target_vocab_size))
    self.encoder_fw_grad_proto = localize(torch.zeros(self.batch_size, self.max_encoder_l, self.encoder_num_hidden))
    self.encoder_bw_grad_proto = localize(torch.zeros(self.batch_size, self.max_encoder_l, self.encoder_num_hidden))
    self.cnn_grad_proto = localize(torch.zeros(self.batch_size, self.max_encoder_l, self.cnn_feature_size))

    local num_params = 0
    self.params, self.grad_params = {}, {}
    for i = 1, #self.layers do
        local p, gp = self.layers[i]:getParameters()
        num_params = num_params + p:size(1)
        self.params[i] = p
        self.grad_params[i] = gp
    end
    log(string.format('Number of parameters: %d', num_params))

    self.encoder_fw_clones = clone_many_times(self.encoder_fw, self.max_encoder_l)
    self.encoder_bw_clones = clone_many_times(self.encoder_bw, self.max_encoder_l)

    -- initalial states
    local encoder_h_init = localize(torch.zeros(self.batch_size, self.encoder_num_hidden))

    self.init_fwd_enc = {}
    self.init_bwd_enc = {}
    for L = 1, self.encoder_num_layers do
        table.insert(self.init_fwd_enc, encoder_h_init:clone())
        table.insert(self.init_fwd_enc, encoder_h_init:clone())
        table.insert(self.init_bwd_enc, encoder_h_init:clone())
        table.insert(self.init_bwd_enc, encoder_h_init:clone())
    end
    for i = 1, #self.encoder_fw_clones do
        if self.encoder_fw_clones[i].apply then
            self.encoder_fw_clones[i]:apply(function(m) m:setReuse() end)
            if self.prealloc then self.encoder_fw_clones[i]:apply(function(m) m:setPrealloc() end) end
        end
    end
    for i = 1, #self.encoder_bw_clones do
        if self.encoder_bw_clones[i].apply then
            self.encoder_bw_clones[i]:apply(function(m) m:setReuse() end)
            if self.prealloc then self.encoder_bw_clones[i]:apply(function(m) m:setPrealloc() end) end
        end
    end
    self.visualize = false
end

-- one step 
function model:step(batch, forward_only, beam_size, trie)
    if forward_only then
        self.trie_locations = {}
    end
    local input_batch = localize(batch[1])
    local num_nonzeros = batch[4]
    local img_paths
    if self.visualize then
        img_paths = batch[5]
    end
    local ctc_sizes = batch[6]
    local ctc_labels = batch[7]

    local batch_size = input_batch:size()[1]

    if not forward_only then
        self.cnn_model:training()
        self.output_projector:training()
    else
        self.cnn_model:evaluate()
        self.output_projector:evaluate()
    end

    local feval = function(p) --cut off when evaluate
        local cnn_output = self.cnn_model:forward(input_batch)
        local source_l = cnn_output:size()[2]
        assert(source_l <= self.max_encoder_l, string.format('max_encoder_l (%d) < source_l (%d)!', self.max_encoder_l, source_l))
        source = cnn_output:transpose(1,2)
        local context = self.context_proto[{{1, batch_size}, {1, source_l}}]
        -- forward encoder
        local rnn_state_enc = reset_state(self.init_fwd_enc, batch_size, 0)
        for t = 1, source_l do
            if not forward_only then
                self.encoder_fw_clones[t]:training()
            else
                self.encoder_fw_clones[t]:evaluate()
            end
            local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
            local out = self.encoder_fw_clones[t]:forward(encoder_input)
            rnn_state_enc[t] = out
            context[{{},t, {1, self.encoder_num_hidden}}]:copy(out[#out])
        end
        local rnn_state_enc_bwd = reset_state(self.init_fwd_enc, batch_size, source_l+1)
        for t = source_l, 1, -1 do
            if not forward_only then
                self.encoder_bw_clones[t]:training()
            else
                self.encoder_bw_clones[t]:evaluate()
            end
            local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
            local out = self.encoder_bw_clones[t]:forward(encoder_input)
            rnn_state_enc_bwd[t] = out
            context[{{},t, {1+self.encoder_num_hidden, 2*self.encoder_num_hidden}}]:copy(out[#out])
        end
        local ctc_acts = self.ctc_acts_proto[{{1, batch_size*source_l}, {}}]
        for t = 1, source_l do
            local probs = self.output_projector:forward(context[{{}, t, {}}]:contiguous():view(batch_size, -1)) -- batch_size, vocab_size
            ctc_acts[{{1+(t-1)*batch_size, t*batch_size}, {}}]:copy(probs)
        end
        local ctc_grads = localize(torch.Tensor())
        if not forward_only then
            ctc_grads = self.ctc_grads_proto[{{1, batch_size*source_l}, {}}]
        end
        local gold_scores = gpu_ctc(ctc_acts, ctc_grads, ctc_labels, ctc_sizes)
        local loss = reduce(gold_scores) / batch_size

        local preds = {}
        local indices
        local rnn_state_dec
        --local loss, accuracy = 0.0, 0.0
        if forward_only then
            -- final decoding
            local labels = {}
            for i = 1, batch_size do
                labels[i] = {}
                local prev_vocab_id = nil
                for j = 1, source_l do
                    local acts = ctc_acts[i+(j-1)*batch_size]
                    local _, vocab_id = torch.max(acts, 1)
                    vocab_id = vocab_id:double()[1]
                    if vocab_id ~= 1 and ( (prev_vocab_id == nil) or (vocab_id ~= prev_vocab_id) ) then
                        labels[i][#labels[i]+1] = vocab_id-1
                    end
                    prev_vocab_id = vocab_id
                end
            end
            local ctc_sizes = {}
            for i = 1, batch_size do
                ctc_sizes[i] = #labels[i]
            end
            local pred_scores = gpu_ctc(ctc_acts, ctc_grads, labels, ctc_sizes)
            local word_err, labels_pred, labels_gold = evalWordErrRate(labels, ctc_labels, self.visualize)
            accuracy = batch_size - word_err
            if self.visualize then
                for i = 1, #img_paths do
                    self.visualize_file:write(string.format('%s\t%s\t%s\t%f\t%f\n', img_paths[i], labels_gold[i], labels_pred[i], -1.0*pred_scores[i], -1.0*gold_scores[i]))
                end
                self.visualize_file:flush()
            end
        else
            local encoder_fw_grads = self.encoder_fw_grad_proto[{{1, batch_size}, {1, source_l}}]
            local encoder_bw_grads = self.encoder_bw_grad_proto[{{1, batch_size}, {1, source_l}}]
            for i = 1, #self.grad_params do
                self.grad_params[i]:zero()
            end
            encoder_fw_grads:zero()
            encoder_bw_grads:zero()
            local ctc_acts = self.ctc_acts_proto[{{1, batch_size*source_l}, {}}]
            for t = 1, source_l do
                local dlst = self.output_projector:backward(context[{{}, t, {}}]:contiguous():view(batch_size, -1), ctc_grads[{{1+(t-1)*batch_size, t*batch_size}, {}}]) -- batch_size, 2*encoder_num_hidden
                dlst:div(batch_size)
                encoder_fw_grads[{{}, t, {}}]:add(dlst[{{}, {1, self.encoder_num_hidden}}])
                encoder_bw_grads[{{}, t, {}}]:add(dlst[{{}, {self.encoder_num_hidden+1, 2*self.encoder_num_hidden}}])
            end
            local cnn_grad = self.cnn_grad_proto[{{1, batch_size}, {1, source_l}, {}}]
            -- forward directional encoder
            local drnn_state_enc = reset_state(self.init_bwd_enc, batch_size)
            local L = self.encoder_num_layers
            for t = source_l, 1, -1 do
                local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
                drnn_state_enc[#drnn_state_enc]:add(encoder_fw_grads[{{},t}])
                local dlst = self.encoder_fw_clones[t]:backward(encoder_input, drnn_state_enc)
                for j = 1, #drnn_state_enc do
                    drnn_state_enc[j]:copy(dlst[j+1])
                end
                cnn_grad[{{}, t, {}}]:copy(dlst[1])
            end
            -- backward directional encoder
            local drnn_state_enc = reset_state(self.init_bwd_enc, batch_size)
            local L = self.encoder_num_layers
            for t = 1, source_l do
                local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
                drnn_state_enc[#drnn_state_enc]:add(encoder_bw_grads[{{},t}])
                local dlst = self.encoder_bw_clones[t]:backward(encoder_input, drnn_state_enc)
                for j = 1, #drnn_state_enc do
                    drnn_state_enc[j]:copy(dlst[j+1])
                end
                cnn_grad[{{}, t, {}}]:add(dlst[1])
            end
            -- cnn
            self.cnn_model:backward(input_batch, cnn_grad)
            collectgarbage()
        end
        return loss, self.grad_params, {num_nonzeros, accuracy}
    end
    local optim_state = self.optim_state
    if not forward_only then
        --local _, loss, stats = optim.adadelta_list(feval, self.params, optim_state); loss = loss[1]
        local _, loss, stats = optim.sgd_list(feval, self.params, optim_state); loss = loss[1]
        return loss*batch_size, stats
    else
        local loss, _, stats = feval(self.params)
        return loss*batch_size, stats -- todo: accuracy
    end
end
-- Set visualize phase
function model:vis(output_dir)
    self.visualize = true
    self.visualize_path = paths.concat(output_dir, 'results.txt')
    local file, err = io.open(self.visualize_path, "w")
    self.visualize_file = file
    if err then 
        log(string.format('Error: visualize file %s cannot be created', self.visualize_path))
        self.visualize  = false
        self.visualize_file = nil
    end
end
-- Save model to model_path
function model:save(model_path)
    for i = 1, #self.layers do
        self.layers[i]:clearState()
    end
    torch.save(model_path, {{self.cnn_model, self.encoder_fw, self.encoder_bw, self.output_projector}, self.config, self.global_step, self.optim_state})
end

function model:shutdown()
    if self.visualize_file then
        self.visualize_file:close()
    end
end
