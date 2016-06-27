 --[[ Training, adapted from https://github.com/harvardnlp/seq2seq-attn/blob/master/train.lua
--]]
require 'nn'
require 'nngraph'
require 'hdf5'
require 'cudnn'
require 'optim'

package.path = package.path .. ";src/?.lua" .. ";src/data_util/?.lua"
require 'cnn'
require 'LSTM'
require 'model_utils'
require 'data_gen'
require 'output_unit'
require 'criterion'
require 'optim_adadelta'
require 'paths'

cmd = torch.CmdLine()

-- Input and Output
cmd:text("")
cmd:text("**Input and Output**")
cmd:text("")
cmd:option('-data_base_dir', '/mnt/90kDICT32px', [[The base directory of the image path in data-path. If the image path in data-path is absolute path, set it to /]])
cmd:option('-data_path', '/mnt/val_shuffled_words.txt', [[The path containing data file names and labels. Format per line: image_path characters]])
cmd:option('-val_data_path', '/mnt/val_shuffled_words.txt', [[The path containing validate data file names and labels. Format per line: image_path characters]])
cmd:option('-model_dir', 'train', [[The directory for saving and loading model parameters (structure is not stored)]])
cmd:option('-log_path', 'log.txt', [[The path to put log]])
cmd:option('-output_dir', 'results', [[The path to put visualization results if visualize is set to True]])
cmd:option('-steps_per_checkpoint', 4, [[Checkpointing (print perplexity, save model) per how many steps]])

-- Optimization
cmd:text("")
cmd:text("**Optimization**")
cmd:text("")
cmd:option('-num_epochs', 1000, [[The number of whole data passes]])
cmd:option('-batch_size', 64, [[Batch size]])
cmd:option('-initial_learning_rate', 0.001, [[Initial learning rate, note the we use AdaDelta, so the initial value doe not matter much]])

-- Network
cmd:option('-dropout', 0.3, [[Dropout probability]])
cmd:option('-target_embedding_size', 10, [[Embedding dimension for each target]])
cmd:option('-attn_use_lstm', 1, [[Whether or not use LSTM attention decoder cell]])
cmd:option('-input_feed', 1, [[Whether or not use LSTM attention decoder cell]])
cmd:option('-attn_num_hidden', 512, [[Number of hidden units in attention decoder cell]])
cmd:option('-encoder_num_hidden', 512, [[Number of hidden units in encoder cell]])
cmd:option('-attn_num_layers', 2, [[Number of layers in attention decoder cell (Encoder number of hidden units will be attn-num-hidden*attn-num-layers)]])
cmd:option('-target_vocab_size', 26+10+3, [[Target vocabulary size. Default is = 26+10+3 # 0: PADDING, 1: GO, 2: EOS, >2: 0-9, a-z]])

-- Other
cmd:option('-phase', 'test', [[train or test]])
cmd:option('-gpuid', 1, [[Which gpu to use. -1 = use CPU]])
cmd:option('-load_model', 1, [[Load model from model-dir or not]])
cmd:option('-seed', 910820, [[Load model from model-dir or not]])
cmd:option('-max_decoder_l', 30, [[Maximum number of output targets]])
cmd:option('-max_encoder_l', 50, [[Maximum length of input feature sequence]])

opt = cmd:parse(arg)
assert (opt.gpuid > 0, 'only supports gpu!')
torch.manualSeed(opt.seed)

function train(train_data, valid_data)
    local num_params = 0
    params, grad_params = {}, {}
    for i = 1, #layers do
        local p, gp = layers[i]:getParameters()
        num_params = num_params + p:size(1)
        params[i] = p
        grad_params[i] = gp
    end
    print("Number of parameters: " .. num_params)
    -- initalize
    local encoder_h_init = torch.zeros(opt.batch_size, opt.encoder_num_hidden)
    local decoder_h_init = torch.zeros(opt.batch_size, opt.attn_num_hidden)
    if opt.gpuid > 0 then
        encoder_h_init = encoder_h_init:cuda()
        decoder_h_init = decoder_h_init:cuda()
    end
    init_fwd_enc = {}
    init_bwd_enc = {}
    init_fwd_dec = {}
    init_bwd_dec = {}
    table.insert(init_fwd_enc, encoder_h_init:clone())
    table.insert(init_fwd_enc, encoder_h_init:clone())
    table.insert(init_bwd_enc, encoder_h_init:clone())
    table.insert(init_bwd_enc, encoder_h_init:clone())
    if opt.input_feed == 1 then
        table.insert(init_fwd_dec, decoder_h_init:clone())
    end
    table.insert(init_bwd_dec, decoder_h_init:clone())
    for L = 1, opt.attn_num_layers do
        table.insert(init_fwd_dec, decoder_h_init:clone()) -- memory cell
        table.insert(init_fwd_dec, decoder_h_init:clone()) -- hidden state
        table.insert(init_bwd_dec, decoder_h_init:clone())
        table.insert(init_bwd_dec, decoder_h_init:clone()) 
    end
    dec_offset = 3 -- offset depends on input feeding
    if opt.input_feed == 1 then
        dec_offset = dec_offset + 1
    end

    -- train
    function step(batch, forward_only)
        local input_batch = batch[1]
        local target_batch = batch[2]
        local batch_size = input_batch:size()[1]
        local decoder_l = target_batch:size()[2]
        cnn_model:training()
        local feval = function(p)
            --if p ~= params then
            --   params:copy(p)
            --end
            output_unit:training()
            if opt.gpuid > 0 then
                input_batch = input_batch:cuda()
                target_batch = target_batch:cuda()
            end
            local cnn_output = cnn_model:forward(input_batch)
            local source_l = cnn_output:size()[2]
            local target_l = target_batch:size()[2]
            source = cnn_output:transpose(1,2)
            target = target_batch:transpose(1,2)
            local context = context_proto[{{1, batch_size}, {1, source_l}}]
            -- forward encoder
            local rnn_state_enc = reset_state(init_fwd_enc, batch_size, 0)
            for t = 1, source_l do
                encoder_fw_clones[t]:training()
                local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
                local out = encoder_fw_clones[t]:forward(encoder_input)
                rnn_state_enc[t] = out
                context[{{},t}]:copy(out[#out])
            end
            local rnn_state_enc_bwd = reset_state(init_fwd_enc, batch_size, source_l+1)
            for t = source_l, 1, -1 do
                encoder_bw_clones[t]:training()
                local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
                local out = encoder_bw_clones[t]:forward(encoder_input)
                rnn_state_enc_bwd[t] = out
                context[{{},t}]:add(out[#out])          
            end
            -- set decoder states
            local rnn_state_dec = reset_state(init_fwd_dec, batch_size, 0)
            for L = 1, opt.attn_num_layers do
                rnn_state_dec[0][L*2-1+opt.input_feed]:copy(rnn_state_enc[source_l][1*2-1]:add(rnn_state_enc_bwd[1][1*2-1]))
                rnn_state_dec[0][L*2+opt.input_feed]:copy(rnn_state_enc[source_l][1*2]:add(rnn_state_enc_bwd[1][1*2]))
            end
            local preds = {}
            local indices
            for t = 1, target_l do
                decoder_clones[t]:training()
                local decoder_input
                if forward_only == 0 or t == 1 then
                    decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
                else
                    decoder_input = {indices, context, table.unpack(rnn_state_dec[t-1])}
                end
                local out = decoder_clones[t]:forward(decoder_input)
                local next_state = {}
                table.insert(preds, out[#out])
                local pred = output_unit:forward(preds[t])
                _, indices = torch.max(pred, 2)
                indices = indices:view(indices:nElement())
                if opt.input_feed == 1 then
                    table.insert(next_state, out[#out])
                end
                for j = 1, #out-1 do
                    table.insert(next_state, out[j])
                end
                rnn_state_dec[t] = next_state
            end
            local loss = 0
            if forward_only == 1 then
                for t = target_l, 1, -1 do
                    local pred = output_unit:forward(preds[t])
                    loss = loss + criterion:forward(pred, target[t])/batch_size
                    _, indices = torch.max(pred, 2)
                end
            else
                local encoder_grads = encoder_grad_proto[{{1, batch_size}, {1, source_l}}]
                local encoder_bwd_grads = encoder_bwd_grad_proto[{{1, batch_size}, {1, source_l}}]
                for i = 1, #grad_params do
                    grad_params[i]:zero()
                end
                encoder_grads:zero()
                encoder_bwd_grads:zero()
                local drnn_state_dec = reset_state(init_bwd_dec, batch_size)
                for t = target_l, 1, -1 do
                    local pred = output_unit:forward(preds[t])
                    loss = loss + criterion:forward(pred, target[t])/batch_size
                    local dl_dpred = criterion:backward(pred, target[t])
                    dl_dpred:div(batch_size)
                    local dl_dtarget = output_unit:backward(preds[t], dl_dpred)
                    drnn_state_dec[#drnn_state_dec]:add(dl_dtarget)
                    local decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
                    local dlst = decoder_clones[t]:backward(decoder_input, drnn_state_dec)
                    encoder_grads:add(dlst[2])
                    encoder_bwd_grads:add(dlst[2])
                    drnn_state_dec[#drnn_state_dec]:zero()
                    if opt.input_feed == 1 then
                        drnn_state_dec[#drnn_state_dec]:add(dlst[3])
                    end     
                    for j = dec_offset, #dlst do
                        drnn_state_dec[j-dec_offset+1]:copy(dlst[j])
                    end
                end
                local drnn_state_enc = reset_state(init_bwd_enc, batch_size)
                for L = 1, opt.attn_num_layers do
                    if L == 1 then
                        drnn_state_enc[1*2-1]:copy(drnn_state_dec[L*2-1])
                        drnn_state_enc[1*2]:copy(drnn_state_dec[L*2])
                    else
                        drnn_state_enc[1*2-1]:add(drnn_state_dec[L*2-1])
                        drnn_state_enc[1*2]:add(drnn_state_dec[L*2])
                    end
                end
                for t = source_l, 1, -1 do
                    local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
                    drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{},t}])
                    local dlst = encoder_fw_clones[t]:backward(encoder_input, drnn_state_enc)
                    for j = 1, #drnn_state_enc do
                        drnn_state_enc[j]:copy(dlst[j+1])
                    end     
                end
                local drnn_state_enc = reset_state(init_bwd_enc, batch_size)
                for L = 1, opt.attn_num_layers do
                    if L == 1 then
                        drnn_state_enc[1*2-1]:copy(drnn_state_dec[L*2-1])
                        drnn_state_enc[1*2]:copy(drnn_state_dec[L*2])
                    else
                        drnn_state_enc[1*2-1]:add(drnn_state_dec[L*2-1])
                        drnn_state_enc[1*2]:add(drnn_state_dec[L*2])
                    end
                end
                for t = 1, source_l do
                    local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
                    drnn_state_enc[#drnn_state_enc]:add(encoder_bwd_grads[{{},t}])
                    local dlst = encoder_bw_clones[t]:backward(encoder_input, drnn_state_enc)
                    for j = 1, #drnn_state_enc do
                        drnn_state_enc[j]:copy(dlst[j+1])
                    end
                end
            end
            return loss, grad_params
        end
        local optim_state = opt.optim_state
        if forward_only == 0 then
            local _, loss = optim.adadelta_list(feval, params, optim_state); loss = loss[1]
            return loss
        else
            loss, _ = feval(params)
            return loss -- todo: accuracy
        end
    end
    local loss = 0
    local num_seen = 0
    for epoch = 1, opt.num_epochs do
        train_data:shuffle()
        local forward_only
        if opt.phase == 'train' then
            forward_only = 0
        else
            forward_only = 1
        end
        while true do
            train_batch = train_data:nextBatch(opt.batch_size)
            if train_batch == nil then
                break
            end
            loss = loss + step(train_batch, forward_only)
            num_seen = num_seen + 1
            print (loss/num_seen)
            num_steps = num_steps + 1
            if num_steps % opt.steps_per_checkpoint == 0 then
                print (string.format('Step %d - train loss = %f', num_steps, loss/num_seen))
                if opt.phase == 'train' then
                    print ('saving model')
                    local savefile = paths.concat(opt.model_dir, string.format('model-%d', num_steps))
                    local finalfile_tmp = paths.concat(opt.model_dir, '.final-model.tmp')
                    local finalfile = paths.concat(opt.model_dir, 'final-model')
                    torch.save(savefile, {{cnn_model, encoder_fw, encoder_bw, decoder, output_unit}, opt, num_steps})
                    os.execute(string.format('cp %s %s', savefile, finalfile_tmp))
                    os.execute(string.format('mv %s %s', finalfile_tmp, finalfile))
                end
                num_seen = 0
                loss = 0
            end
        end
    end

end
function main() 
    -- Parse command line 
    opt = cmd:parse(arg)
    n = opt.attn_num_layers
    input_size = 512

    if opt.gpuid >= 0 then
      print ('Using CUDA on GPU ' .. opt.gpuid .. '...')
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(opt.gpuid)
      cutorch.manualSeed(opt.seed)      
    end
    -- Load data
    -- Build model
    if opt.load_model == 0 or not paths.filep(paths.concat(opt.model_dir, 'final-model')) then
        cnn_model = createCNNModel()
        encoder_fw = createLSTM(input_size, opt.encoder_num_hidden, 1, 0, 0, opt.dropout, 0, 0)
        encoder_bw = createLSTM(input_size, opt.encoder_num_hidden, 1, 0, 0, opt.dropout, 0, 0)
        decoder = createLSTM(2*opt.encoder_num_hidden, opt.attn_num_hidden, opt.attn_num_layers, 1, 1, opt.dropout, 1, opt.target_vocab_size)
        output_unit = createOutputUnit(opt.attn_num_hidden, opt.target_vocab_size)
        num_steps = 0

    else
        print('loading model')
        local checkpoint = torch.load(paths.concat(opt.model_dir, 'final-model'))
        local model, model_opt = checkpoint[1], checkpoint[2]
        cnn_model = model[1]:double()
        encoder_fw = model[2]:double()
        encoder_bw = model[3]:double()
        decoder = model[4]:double()      
        output_unit = model[5]:double()
        num_steps = checkpoint[3]
    end
    criterion = createCriterion(opt.target_vocab_size)

    context_proto = torch.zeros(opt.batch_size, opt.max_encoder_l, opt.encoder_num_hidden)
    encoder_grad_proto = torch.zeros(opt.batch_size, opt.max_encoder_l, opt.encoder_num_hidden)
    encoder_bwd_grad_proto = torch.zeros(opt.batch_size, opt.max_encoder_l, opt.encoder_num_hidden)
    layers = {cnn_model, encoder_fw, encoder_bw, decoder, output_unit}
    if opt.gpuid >= 0 then
        for i = 1, #layers do
            layers[i]:cuda()
        end
        criterion:cuda()
        context_proto = context_proto:cuda()
        encoder_grad_proto = encoder_grad_proto:cuda()
        encoder_bwd_grad_proto = encoder_bwd_grad_proto:cuda()
    end
    decoder_clones = clone_many_times(decoder, opt.max_decoder_l)
    encoder_fw_clones = clone_many_times(encoder_fw, opt.max_encoder_l)
    encoder_bw_clones = clone_many_times(encoder_bw, opt.max_encoder_l)
    print(string.format('Load training data from %s', opt.data_path))
    local train_data = DataGen(opt.data_base_dir, opt.data_path)
    print(string.format('Load validating data from %s', opt.val_data_path))
    local val_data = DataGen(opt.data_base_dir, opt.val_data_path)
    opt.optim_state = {}

    if not paths.dirp(opt.model_dir) then
        paths.mkdir(opt.model_dir)
    end
    train(train_data, val_data)
end

main()
