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
require 'optim_adadelta'

local model = torch.class('Model')

--[[ Args: 
-- config.load_model
-- config.model_dir
-- config.dropout
-- config.encoder_num_hidden
-- config.encoder_num_layers
-- config.decoder_num_layers
-- config.target_vocab_size
-- config.target_embedding_size
-- config.max_encoder_l
-- config.max_decoder_l
-- config.input_feed
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

    log(string.format('Loading model from %s', model_path))
    local checkpoint = torch.load(model_path)
    local model, model_config = checkpoint[1], checkpoint[2]
    self.cnn_model = model[1]:double()
    self.encoder_fw = model[2]:double()
    self.encoder_bw = model[3]:double()
    self.decoder = model[4]:double()      
    self.output_projector = model[5]:double()
    self.global_step = checkpoint[3]
    self.optim_state = checkpoint[4]

    print(model_config)
    -- Load model structure parameters
    self.cnn_feature_size = 512
    self.dropout = model_config.dropout
    self.encoder_num_hidden = model_config.encoder_num_hidden
    self.encoder_num_layers = model_config.encoder_num_layers
    self.decoder_num_hidden = self.encoder_num_hidden * 2
    self.decoder_num_layers = model_config.decoder_num_layers
    self.target_vocab_size = model_config.target_vocab_size
    self.target_embedding_size = model_config.target_embedding_size
    self.input_feed = config.input_feed

    self.max_encoder_l = config.max_encoder_l or model_config.max_encoder_l
    self.max_decoder_l = config.max_decoder_l or model_config.max_decoder_l
    self.batch_size = config.batch_size or model_config.batch_size
    self:_build()
end

-- create model with fresh parameters
function model:create(config)
    self.cnn_feature_size = 512
    self.dropout = config.dropout
    self.encoder_num_hidden = config.encoder_num_hidden
    self.encoder_num_layers = config.encoder_num_layers
    self.decoder_num_hidden = config.encoder_num_hidden * 2
    self.decoder_num_layers = config.decoder_num_layers
    self.target_vocab_size = config.target_vocab_size
    self.target_embedding_size = config.target_embedding_size
    self.max_encoder_l = config.max_encoder_l
    self.max_decoder_l = config.max_decoder_l
    self.input_feed = config.input_feed
    self.batch_size = config.batch_size

    -- CNN model, input size: (batch_size, 1, 32, width), output size: (batch_size, sequence_length, 512)
    self.cnn_model = createCNNModel()
    -- createLSTM(input_size, num_hidden, num_layers, dropout, use_attention, input_feed, use_lookup, vocab_size)
    self.encoder_fw = createLSTM(self.cnn_feature_size, self.encoder_num_hidden, self.encoder_num_layers, self.dropout, false, false, false)
    self.encoder_bw = createLSTM(self.cnn_feature_size, self.encoder_num_hidden, self.encoder_num_layers, self.dropout, false, false, false)
    self.decoder = createLSTM(self.target_embedding_size, self.decoder_num_hidden, self.decoder_num_layers, self.dropout, true, self.input_feed, true, self.target_vocab_size)
    self.output_projector = createOutputUnit(self.decoder_num_hidden, self.target_vocab_size)
    self.global_step = 0

    self:_build()
end

-- build
function model:_build()
    log(string.format('cnn_featuer_size: %d', self.cnn_feature_size))
    log(string.format('dropout: %f', self.dropout))
    log(string.format('encoder_num_hidden: %d', self.encoder_num_hidden))
    log(string.format('encoder_num_layers: %d', self.encoder_num_layers))
    log(string.format('decoder_num_hidden: %d', self.decoder_num_hidden))
    log(string.format('decoder_num_layers: %d', self.decoder_num_layers))
    log(string.format('target_vocab_size: %d', self.target_vocab_size))
    log(string.format('target_embedding_size: %d', self.target_embedding_size))
    log(string.format('max_encoder_l: %d', self.max_encoder_l))
    log(string.format('max_decoder_l: %d', self.max_decoder_l))
    log(string.format('input_feed: %s', self.input_feed))
    log(string.format('batch_size: %d', self.batch_size))

    self.config = {}
    self.config.dropout = self.dropout
    self.config.encoder_num_hidden = self.encoder_num_hidden
    self.config.encoder_num_layers = self.encoder_num_layers
    self.config.decoder_num_hidden = self.decoder_num_hidden
    self.config.decoder_num_layers = self.decoder_num_layers
    self.config.target_vocab_size = self.target_vocab_size
    self.config.target_embedding_size = self.target_embedding_size
    self.config.max_encoder_l = self.max_encoder_l
    self.config.max_decoder_l = self.max_decoder_l
    self.config.input_feed = self.input_feed
    self.config.batch_size = self.batch_size

    if self.optim_state == nil then
        self.optim_state = {}
    end
    self.criterion = createCriterion(self.target_vocab_size)

    -- convert to cuda if use gpu
    self.layers = {self.cnn_model, self.encoder_fw, self.encoder_bw, self.decoder, self.output_projector}
    for i = 1, #self.layers do
        localize(self.layers[i])
    end
    localize(self.criterion)

    self.decoder_clones = clone_many_times(self.decoder, self.max_decoder_l)
    self.encoder_fw_clones = clone_many_times(self.encoder_fw, self.max_encoder_l)
    self.encoder_bw_clones = clone_many_times(self.encoder_bw, self.max_encoder_l)

    self.context_proto = localize(torch.zeros(self.batch_size, self.max_encoder_l, 2*self.encoder_num_hidden))
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

    -- initalial states
    local encoder_h_init = localize(torch.zeros(self.batch_size, self.encoder_num_hidden))
    local decoder_h_init = localize(torch.zeros(self.batch_size, self.decoder_num_hidden))

    self.init_fwd_enc = {}
    self.init_bwd_enc = {}
    self.init_fwd_dec = {}
    self.init_bwd_dec = {}
    for L = 1, self.encoder_num_layers do
        table.insert(self.init_fwd_enc, encoder_h_init:clone())
        table.insert(self.init_fwd_enc, encoder_h_init:clone())
        table.insert(self.init_bwd_enc, encoder_h_init:clone())
        table.insert(self.init_bwd_enc, encoder_h_init:clone())
    end
    if self.input_feed then
        table.insert(self.init_fwd_dec, decoder_h_init:clone())
    end
    table.insert(self.init_bwd_dec, decoder_h_init:clone())
    for L = 1, self.decoder_num_layers do
        table.insert(self.init_fwd_dec, decoder_h_init:clone()) -- memory cell
        table.insert(self.init_fwd_dec, decoder_h_init:clone()) -- hidden state
        table.insert(self.init_bwd_dec, decoder_h_init:clone())
        table.insert(self.init_bwd_dec, decoder_h_init:clone()) 
    end
    self.dec_offset = 3 -- offset depends on input feeding
    if self.input_feed then
        self.dec_offset = self.dec_offset + 1
    end
    self.init_beam = false
end

-- one step 
function model:step(batch, forward_only, beam_size)
    beam_size = beam_size or 1 -- default argmax
    if forward_only then
        if not self.init_beam then
            self.init_beam = true
            local beam_decoder_h_init = localize(torch.zeros(self.batch_size*beam_size, self.encoder_num_hidden))
            self.beam_scores = localize(torch.zeros(self.batch_size, beam_size))
            self.current_indices_history = {}
            self.beam_parents_history = {}
            self.beam_init_fwd_dec = {}
            if self.input_feed then
                table.insert(self.beam_init_fwd_dec, beam_decoder_h_init:clone())
            end
            for L = 1, self.decoder_num_layers do
                table.insert(self.beam_init_fwd_dec, beam_decoder_h_init:clone()) -- memory cell
                table.insert(self.beam_init_fwd_dec, beam_decoder_h_init:clone()) -- hidden state
            end
        end
    end
    local input_batch = localize(batch[1])
    local target_batch = localize(batch[2])
    local target_eval_batch = localize(batch[3])
    local num_nonzeros = batch[4]

    local batch_size = input_batch:size()[1]
    local target_l = target_batch:size()[2]

    assert(target_l <= self.max_decoder_l, string.format('max_decoder_l (%d) < target_l (%d)!', self.max_decoder_l, target_l))
    -- if forward only, then re-generate the target batch
    if forward_only then
        local target_batch_new = localize(torch.IntTensor(batch_size, self.max_decoder_l)):fill(1)
        target_batch_new[{{1,batch_size}, {1,target_l}}]:copy(target_batch)
        target_batch = target_batch_new
        local target_eval_batch_new = localize(torch.IntTensor(batch_size, self.max_decoder_l)):fill(1)
        target_eval_batch_new[{{1,batch_size}, {1,target_l}}]:copy(target_eval_batch)
        target_eval_batch = target_eval_batch_new
        target_l = self.max_decoder_l
    end

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
        target = target_batch:transpose(1,2)
        target_eval = target_eval_batch:transpose(1,2)
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
        local preds = {}
        local indices
        -- forward_only == true, beam search
        if forward_only then
            local beam_replicate = function(hidden_state)
                if hidden_state:dim() == 1 then
                    local batch_size = hidden_state:size()[1]
                    return torch.view(hidden_state, batch_size, 1):expand(batch_size, beam_size):view(batch_size*beam_size)
                elseif hidden_state:dim() == 2 then
                    local batch_size = hidden_state:size()[1]
                    local num_hidden = hidden_state:size()[2]
                    return torch.view(hidden_state, batch_size, 1, num_hidden):expand(batch_size, beam_size, num_hidden):view(batch_size*beam_size, num_hidden)
                elseif hidden_state:dim() == 3 then
                    local batch_size = hidden_state:size()[1]
                    local source_l = hidden_state:size()[2]
                    local num_hidden = hidden_state:size()[3]
                    return torch.view(hidden_state, batch_size, 1, source_l, num_hidden):expand(batch_size, beam_size, source_l, num_hidden):view(batch_size*beam_size, source_l, num_hidden)
                else
                    assert(false, 'does not support ndim except for 2 and 3')
                end
            end
            local rnn_state_dec = reset_state(self.beam_init_fwd_dec, batch_size*beam_size, 0)
            if self.input_feed then
                rnn_state_dec[0][1*2-1+1]:copy(beam_replicate(torch.cat(rnn_state_enc[source_l][1*2-1], rnn_state_enc_bwd[1][1*2-1])))
                rnn_state_dec[0][1*2+1]:copy(beam_replicate(torch.cat(rnn_state_enc[source_l][1*2], rnn_state_enc_bwd[1][1*2])))
            else
                rnn_state_dec[0][1*2-1+0]:copy(beam_replicate(torch.cat(rnn_state_enc[source_l][1*2-1], rnn_state_enc_bwd[1][1*2-1])))
                rnn_state_dec[0][1*2+0]:copy(beam_replicate(torch.cat(rnn_state_enc[source_l][1*2], rnn_state_enc_bwd[1][1*2])))
            end
            for L = 2, self.decoder_num_layers do
                rnn_state_dec[0][L*2-1+0]:zero()
                rnn_state_dec[0][L*2+0]:zero()
            end
            context = beam_replicate(context)
            for t = 1, target_l do
                self.decoder_clones[t]:evaluate()
                local decoder_input
                if t ~= 1 then
                    decoder_input = {indices, context, table.unpack(rnn_state_dec[t-1])}
                else
                    decoder_input = {beam_replicate(target[t]), context, table.unpack(rnn_state_dec[t-1])}
                end
                local out = self.decoder_clones[t]:forward(decoder_input)
                local next_state = {}
                local top_out = out[#out]
                local probs = self.output_projector:forward(top_out) -- batch_size*beam_size, vocab_size
                probs[{{}, 1}]:fill(0) -- beam_scores: batch_size, beam_size
                local total_scores = (probs:view(batch_size, beam_size, self.target_vocab_size) + self.beam_scores:view(batch_size, beam_size, 1):expand(batch_size, beam_size, self.target_vocab_size)):view(batch_size, beam_size*target_vocab_size) -- batch_size, beam_size * target_vocab_size
                self.beam_scores, raw_indices = total_scores:topk(beam_size, true) --batch_size, beam_size
                local current_indices = raw_indices:mod(beam_size)+1 -- batch_size, beam_size for current vocab
                local beam_parents = raw_indices:remainder(beam_size)+1 -- batch_size, beam_size for number of beam in each batch
                table.insert(self.current_indices_history, current_indices:clone())
                table.insert(self.beam_parents_history, beam_parents:clone())

                if self.input_feed then
                    local top_out = out[#out] -- batch_size*beam_size, hidden_dim
                    table.insert(next_state, top_out:index(1, beam_parents:view(-1)+torch.range(0,batch_size-1):view(batch_size,1):expand(batch_size,beam_size):view(-1)))
                end
                for j = 1, #out-1 do
                    local out_j = out[j] -- batch_size*beam_size, hidden_dim
                    table.insert(next_state, out_j:index(1, beam_parents:view(-1)+torch.range(0,batch_size-1):view(batch_size,1):expand(batch_size,beam_size):view(-1)))
                end
                rnn_state_dec[t] = next_state
            end
            -- final decoding
            local labels = localize(torch.zeros(batch_size, target_l)):fill(1)
            _, indices = torch.max(self.beam_scores, 
            for t = target_l, 1, -1 do
                _, indices = torch.max(pred, 2)
            end
        else -- forward_only == false
            -- set decoder states
            local rnn_state_dec = reset_state(self.init_fwd_dec, batch_size, 0)
            -- only use encoder final state to initialize the first layer
            if self.input_feed then
                rnn_state_dec[0][1*2-1+1]:copy(torch.cat(rnn_state_enc[source_l][1*2-1], rnn_state_enc_bwd[1][1*2-1]))
                rnn_state_dec[0][1*2+1]:copy(torch.cat(rnn_state_enc[source_l][1*2], rnn_state_enc_bwd[1][1*2]))
            else
                rnn_state_dec[0][1*2-1+0]:copy(torch.cat(rnn_state_enc[source_l][1*2-1], rnn_state_enc_bwd[1][1*2-1]))
                rnn_state_dec[0][1*2+0]:copy(torch.cat(rnn_state_enc[source_l][1*2], rnn_state_enc_bwd[1][1*2]))
            end
            for L = 2, self.decoder_num_layers do
                rnn_state_dec[0][L*2-1+0]:zero()
                rnn_state_dec[0][L*2+0]:zero()
            end
            for t = 1, target_l do
                self.decoder_clones[t]:training()
                local decoder_input
                if forward_only and t ~= 1 then
                    decoder_input = {indices, context, table.unpack(rnn_state_dec[t-1])}
                else
                    decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
                end
                local out = self.decoder_clones[t]:forward(decoder_input)
                local next_state = {}
                table.insert(preds, out[#out])
                local pred = self.output_projector:forward(preds[t])
                _, indices = torch.max(pred, 2)
                indices = indices:view(indices:nElement())
                if self.input_feed then
                    table.insert(next_state, out[#out])
                end
                for j = 1, #out-1 do
                    table.insert(next_state, out[j])
                end
                rnn_state_dec[t] = next_state
            end
        end
        local loss, accuracy = 0.0, 0.0
        if forward_only then
            local labels = localize(torch.zeros(batch_size, target_l)):fill(1)
            for t = 1, target_l do
                local pred = self.output_projector:forward(preds[t])
                loss = loss + self.criterion:forward(pred, target_eval[t])/batch_size
                local _, indices = torch.max(pred, 2)
                indices = indices:view(indices:nElement())
                labels[{{1,batch_size}, t}]:copy(indices)
            end
            accuracy = 1.0 - evalWordErrRate(labels, target_eval_batch)
        else
            local encoder_fw_grads = self.encoder_fw_grad_proto[{{1, batch_size}, {1, source_l}}]
            local encoder_bw_grads = self.encoder_bw_grad_proto[{{1, batch_size}, {1, source_l}}]
            for i = 1, #self.grad_params do
                self.grad_params[i]:zero()
            end
            encoder_fw_grads:zero()
            encoder_bw_grads:zero()
            local drnn_state_dec = reset_state(self.init_bwd_dec, batch_size)
            for t = target_l, 1, -1 do
                local pred = self.output_projector:forward(preds[t])
                loss = loss + self.criterion:forward(pred, target_eval[t])/batch_size
                local dl_dpred = self.criterion:backward(pred, target_eval[t])
                dl_dpred:div(batch_size)
                local dl_dtarget = self.output_projector:backward(preds[t], dl_dpred)
                drnn_state_dec[#drnn_state_dec]:add(dl_dtarget)
                local decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
                local dlst = self.decoder_clones[t]:backward(decoder_input, drnn_state_dec)
                encoder_fw_grads:add(dlst[2][{{}, {}, {1,self.encoder_num_hidden}}])
                encoder_bw_grads:add(dlst[2][{{}, {}, {self.encoder_num_hidden+1, 2*self.encoder_num_hidden}}])
                drnn_state_dec[#drnn_state_dec]:zero()
                if self.input_feed then
                    drnn_state_dec[#drnn_state_dec]:copy(dlst[3])
                end     
                for j = self.dec_offset, #dlst do
                    drnn_state_dec[j-self.dec_offset+1]:copy(dlst[j])
                end
            end
            local cnn_grad = self.cnn_grad_proto[{{1, batch_size}, {1, source_l}, {}}]
            -- forward directional encoder
            local drnn_state_enc = reset_state(self.init_bwd_enc, batch_size)
            local L = self.encoder_num_layers
            drnn_state_enc[L*2-1]:copy(drnn_state_dec[1*2-1][{{}, {1, self.encoder_num_hidden}}])
            drnn_state_enc[L*2]:copy(drnn_state_dec[1*2][{{}, {1, self.encoder_num_hidden}}])
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
            drnn_state_enc[L*2-1]:copy(drnn_state_dec[1*2-1][{{}, {self.encoder_num_hidden+1, 2*self.encoder_num_hidden}}])
            drnn_state_enc[L*2]:copy(drnn_state_dec[1*2][{{}, {self.encoder_num_hidden+1, 2*self.encoder_num_hidden}}])
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
        local _, loss, stats = optim.adadelta_list(feval, self.params, optim_state); loss = loss[1]
        return loss*batch_size, stats
    else
        local loss, _, stats = feval(self.params)
        return loss*batch_size, stats -- todo: accuracy
    end
end

-- Save model to model_path
function model:save(model_path)
    torch.save(model_path, {{self.cnn_model, self.encoder_fw, self.encoder_bw, self.decoder, self.output_projector}, self.config, self.global_step, self.optim_state})
end
