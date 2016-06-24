 --[[ Training, adapted from https://github.com/harvardnlp/seq2seq-attn/blob/master/train.lua
--]]
require 'nn'
require 'nngraph'
require 'hdf5'
require 'cudnn'

package.path = package.path .. ";src/?.lua"
require 'cnn'
require 'src/LSTM.lua'
require 'src/model_utils.lua'

cmd = torch.CmdLine()

-- Input and Output
cmd:text("")
cmd:text("**Input and Output**")
cmd:text("")
cmd:option('-data_base_dir', '/mnt/90kDICT32px', [[The base directory of the image path in data-path. If the image path in data-path is absolute path, set it to /]])
cmd:option('-data_path', '/mnt/train_shuffled_words.txt', [[The path containing data file names and labels. Format per line: image_path characters]])
cmd:option('-model_dir', 'train', [[The directory for saving and loading model parameters (structure is not stored)]])
cmd:option('-log_path', 'log.txt', [[The path to put log]])
cmd:option('-output_dir', 'results', [[The path to put visualization results if visualize is set to True]])
cmd:option('-steps_per_checkpoint', 400, [[Checkpointing (print perplexity, save model) per how many steps]])

-- Optimization
cmd:text("")
cmd:text("**Optimization**")
cmd:text("")
cmd:option('-num_epoch', 1000, [[The number of whole data passes]])
cmd:option('-batch_size', 64, [[Batch size]])
cmd:option('-initial_learning_rate', 0.001, [[Initial learning rate, note the we use AdaDelta, so the initial value doe not matter much]])

-- Network
cmd:option('-dropout', 0.3, [[Dropout probability]])
cmd:option('-target_embedding_size', 10, [[Embedding dimension for each target]])
cmd:option('-target_embedding_size', 10, [[Embedding dimension for each target]])
cmd:option('-attn_use_lstm', 1, [[Whether or not use LSTM attention decoder cell]])
cmd:option('-attn_num_hidden', 128, [[Number of hidden units in attention decoder cell]])
cmd:option('-attn_num_layers', 2, [[Number of layers in attention decoder cell (Encoder number of hidden units will be attn-num-hidden*attn-num-layers)]])
cmd:option('-target_vocag_size', 26+10+3, [[Target vocabulary size. Default is = 26+10+3 # 0: PADDING, 1: GO, 2: EOS, >2: 0-9, a-z]])

-- Other
cmd:option('-gpuid', -1, [[Which gpu to use. -1 = use CPU]])
cmd:option('-load_model', 1, [[Load model from model-dir or not]])
cmd:option('-seed', 910820, [[Load model from model-dir or not]])

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

function main() 
    -- Parse command line 
    opt = cmd:parse(arg)
    n = opt.attn_num_layers
    rnn_size = opt.attn_num_hidden
    input_size = 512
    dropout = opt.dropout

    if opt.gpuid >= 0 then
      print ('Using CUDA on GPU ' .. opt.gpuid .. '...')
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(opt.gpuid)
      cutorch.manualSeed(opt.seed)      
    end
    -- Load data
    -- Build model
    cnn = createCNNModel()
    encoder = create_lstm(input_size, rnn_size, n, 0, 0, dropout)
    decoder = create_lstm(input_size, rnn_size, n, 1, 1, dropout)

    layers = {cnn, encoder, decoder}
    if opt.gpuid >= 0 then
        for i = 1, #layers do
            layers[i]:cuda()
        end
    end
    train(train_data, valid_data)
end

main()
