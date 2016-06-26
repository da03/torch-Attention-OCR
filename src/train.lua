 --[[ Training, adapted from https://github.com/harvardnlp/seq2seq-attn/blob/master/train.lua
--]]
require 'nn'
require 'nngraph'
require 'hdf5'
require 'cudnn'

package.path = package.path .. ";src/?.lua" .. ";src/data_util/?.lua"
require 'cnn'
require 'LSTM'
require 'model_utils'
require 'data_gen'

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
cmd:option('-steps_per_checkpoint', 400, [[Checkpointing (print perplexity, save model) per how many steps]])

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
cmd:option('-target_embedding_size', 10, [[Embedding dimension for each target]])
cmd:option('-attn_use_lstm', 1, [[Whether or not use LSTM attention decoder cell]])
cmd:option('-attn_num_hidden', 128, [[Number of hidden units in attention decoder cell]])
cmd:option('-encoder_num_hidden', 127, [[Number of hidden units in encoder cell]])
cmd:option('-attn_num_layers', 2, [[Number of layers in attention decoder cell (Encoder number of hidden units will be attn-num-hidden*attn-num-layers)]])
cmd:option('-target_vocag_size', 26+10+3, [[Target vocabulary size. Default is = 26+10+3 # 0: PADDING, 1: GO, 2: EOS, >2: 0-9, a-z]])

-- Other
cmd:option('-gpuid', -1, [[Which gpu to use. -1 = use CPU]])
cmd:option('-load_model', 1, [[Load model from model-dir or not]])
cmd:option('-seed', 910820, [[Load model from model-dir or not]])
cmd:option('-max_decoder_l', 30, [[Maximum number of output targets]])
cmd:option('-max_encoder_l', 50, [[Maximum length of input feature sequence]])

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

function train(train_data, valid_data)
    local num_params = 0
    params, grad_params = {}, {}
    for i = 1, #layers do
        print (i)
        local p, gp = layers[i]:getParameters()
        --p:uniform(-opt.param_init, opt.param_init)
        num_params = num_params + p:size(1)
        params[i] = p
        grad_params[i] = gp
    end
    print("Number of parameters: " .. num_params)
    -- initalize
    local encoder_h_init = torch.zeros(opt.batch_size, opt.encoder_num_hidden)
    local decoder_h_init = torch.zeros(opt.batch_size, opt.attn_num_hidden)
    if opt.gpuid > 0 then
        encoder_h_init:cuda()
        decoder_h_init:cuda()
    end
    init_fwd_enc = {}
    init_bwd_enc = {}
    init_fwd_dec = {}
    table.insert(init_fwd_enc, encoder_h_init:clone())
    table.insert(init_fwd_enc, encoder_h_init:clone())
    table.insert(init_bwd_enc, encoder_h_init:clone())
    table.insert(init_bwd_enc, encoder_h_init:clone())
    if opt.input_feed == 1 then
        table.insert(init_fwd_dec, decoder_h_init:clone())
    end
    for L = 1, opt.attn_num_layers do
        table.insert(init_fwd_dec, decoder_h_init:clone()) -- memory cell
        table.insert(init_fwd_dec, decoder_h_init:clone()) -- hidden state
    end
    dec_offset = 3 -- offset depends on input feeding
    if opt.input_feed == 1 then
        dec_offset = dec_offset + 1
    end

    -- train
    function step(input_batch, target_batch, forward_only)
        cnn_model:training()
        local feval = function(p)
            --gradParams:zero()
            local cnn_output = cnn_model:forward(inputBatch)
            -- forward encoder
            for t = 1, source_l do
                encoder_clones[t]:training()
                local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
                local out = encoder_clones[t]:forward(encoder_input)
                rnn_state_enc[t] = out
                context[{{},t}]:copy(out[#out])
            end
            -- set decoder states
            for L = 1, opt.num_layers do
                rnn_state_dec[0][L*2-1+opt.input_feed]:copy(rnn_state_enc[source_l][L*2-1])
                rnn_state_dec[0][L*2+opt.input_feed]:copy(rnn_state_enc[source_l][L*2])
            end
            -- backward encoder
            for t = source_l, 1, -1 do
                encoder_bwd_clones[t]:training()
                local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
                local out = encoder_bwd_clones[t]:forward(encoder_input)
                rnn_state_enc_bwd[t] = out
                context[{{},t}]:add(out[#out])          
            end
            for L = 1, opt.attn_num_layers do
                rnn_state_dec[0][L*2-1+opt.input_feed]:add(rnn_state_enc_bwd[1][L*2-1])
                rnn_state_dec[0][L*2+opt.input_feed]:add(rnn_state_enc_bwd[1][L*2])
            end
            for t = 1, target_l do
                decoder_clones[t]:training()
                decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
                local out = decoder_clones[t]:forward(decoder_input)
                local next_state = {}
                table.insert(preds, out[#out])
                if opt.input_feed == 1 then
                    table.insert(next_state, out[#out])
                end
                for j = 1, #out-1 do
                    table.insert(next_state, out[j])
                end
                rnn_state_dec[t] = next_state
            end
            local f = criterion:forward(outputBatch, targetBatch)
            model:backward(inputBatch, criterion:backward(outputBatch, targetBatch))
            gradParams:div(nFrame)
            f = f / nFrame
            return f, gradParams
        end
    end
    local loss
    local num_steps = 0
    for epoch = 1, opt.num_epochs do
        train_data:shuffle()
        for batch_id = 1, train_data:size()/opt.batch_size do
            train_batch = train_data:nextBatch()
            loss = step(train_batch, 0)
            num_steps = num_steps + 1
            if iteratations % opt.steps_per_checkpoint == 0 then
                logging(string.format('Step %d - train loss = %f', num_steps, loss))
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
    cnn_model = createCNNModel()
    encoder_fw = createLSTM(input_size, opt.encoder_num_hidden, 1, 0, 0, opt.dropout)
    encoder_bw = createLSTM(input_size, opt.encoder_num_hidden, 1, 0, 0, opt.dropout)
    decoder = createLSTM(2*opt.encoder_num_hidden, opt.attn_num_hidden, n, 1, 1, opt.dropout)

    decoder_clones = clone_many_times(decoder, opt.max_decoder_l)
    encoder_fw_clones = clone_many_times(encoder_fw, opt.max_encoder_l)
    encoder_bw_clones = clone_many_times(encoder_bw, opt.max_encoder_l)

    layers = {cnn_model, encoder_fw, encoder_bw, decoder}
    if opt.gpuid >= 0 then
        for i = 1, #layers do
            layers[i]:cuda()
        end
    end
    print(string.format('Load training data from %s', opt.data_path))
    local train_data = DataGen(opt.data_base_dir, opt.data_path)
    print(string.format('Load validating data from %s', opt.val_data_path))
    local val_data = DataGen(opt.data_base_dir, opt.val_data_path)
    train(train_data, val_data)
end

main()
