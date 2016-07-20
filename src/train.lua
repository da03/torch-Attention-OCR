 --[[ Training, adapted from https://github.com/harvardnlp/seq2seq-attn/blob/master/train.lua
--]]
torch.setheaptracking(true)
require 'nn'
require 'nngraph'
require 'hdf5'
require 'cudnn'
require 'optim'
require 'paths'

package.path = package.path .. ';src/?.lua' .. ';src/data/?.lua' .. ';src/utils/?.lua' .. ';src/model/?.lua'
require 'model'
require 'data_gen'
require 'logging'
cmd = torch.CmdLine()

-- Input and Output
cmd:text('')
cmd:text('**Input and Output**')
cmd:text('')
cmd:option('-data_base_dir', '/n/rush_lab/data/image_data/90kDICT32px', [[The base directory of the image path in data-path. If the image path in data-path is absolute path, set it to /]])
cmd:option('-data_path', '/n/rush_lab/data/image_data/train_shuffled_shuffled_words.txt', [[The path containing data file names and labels. Format per line: image_path characters]])
cmd:option('-val_data_path', '/n/rush_lab/data/image_data/val_shuffled_words.txt', [[The path containing validate data file names and labels. Format per line: image_path characters]])
cmd:option('-model_dir', 'train', [[The directory for saving and loading model parameters (structure is not stored)]])
cmd:option('-log_path', 'log.txt', [[The path to put log]])
cmd:option('-output_dir', 'results', [[The path to put visualization results if visualize is set to True]])

-- Display
cmd:option('-steps_per_checkpoint', 400, [[Checkpointing (print perplexity, save model) per how many steps]])
cmd:option('-num_batches_val', math.huge, [[Number of batches to evaluate.]])
cmd:option('-beam_size', 1, [[Beam size.]])

-- Optimization
cmd:text('')
cmd:text('**Optimization**')
cmd:text('')
cmd:option('-num_epochs', 1000, [[The number of whole data passes]])
cmd:option('-batch_size', 64, [[Batch size]])

-- Network
cmd:option('-dropout', 0.3, [[Dropout probability]])
cmd:option('-target_embedding_size', 20, [[Embedding dimension for each target]])
cmd:option('-input_feed', false, [[Whether or not use LSTM attention decoder cell]])
cmd:option('-encoder_num_hidden', 512, [[Number of hidden units in encoder cell]])
cmd:option('-encoder_num_layers', 1, [[Number of hidden layers in encoder cell]])
cmd:option('-decoder_num_layers', 2, [[Number of hidden units in decoder cell]])
cmd:option('-target_vocab_size', 26+10+3, [[Target vocabulary size. Default is = 26+10+3 # 1: PADDING, 2: GO, 3: EOS, >3: 0-9, a-z]])

-- Other
cmd:option('-phase', 'test', [[train or test]])
cmd:option('-gpu_id', 1, [[Which gpu to use. <=0 means use CPU]])
cmd:option('-load_model', false, [[Load model from model-dir or not]])
cmd:option('-seed', 910820, [[Load model from model-dir or not]])
cmd:option('-max_decoder_l', 50, [[Maximum number of output targets]]) -- when evaluate, this is the cut-off length.
cmd:option('-max_encoder_l', 80, [[Maximum length of input feature sequence]]) --320*10/4-1

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

function train(model, phase, batch_size, num_epochs, train_data, val_data, model_dir, steps_per_checkpoint, num_batches_val, beam_size)
    local loss = 0
    local num_seen = 0
    local num_samples = 0
    local num_nonzeros = 0
    local accuracy = 0
    local forward_only
    if phase == 'train' then
        forward_only = false
    elseif phase == 'test' then
        forward_only = true
        num_epochs = 1
        model.global_step = 0
    else
        assert(false, 'phase must be either train or test')
    end
    for epoch = 1, num_epochs do
        if not forward_only then
            train_data:shuffle()
            train_data:shuffle()
            train_data:shuffle()
            train_data:shuffle()
        end
        while true do
            train_batch = train_data:nextBatch(batch_size)
            if train_batch == nil then
                break
            end
            local real_batch_size = train_batch[1]:size()[1]
            local step_loss, stats = model:step(train_batch, forward_only, beam_size)
            num_seen = num_seen + 1
            num_samples = num_samples + real_batch_size
            num_nonzeros = num_nonzeros + stats[1]
            if forward_only then
                --print (train_batch[1]:size(4))
                accuracy = accuracy + stats[2]
                --print (stats[2]/real_batch_size)
            else
                loss = loss + step_loss
            end
            --print (loss/num_seen)
            model.global_step = model.global_step + 1
            if model.global_step % steps_per_checkpoint == 0 then
                if forward_only then
                    logging:info(string.format('Number of samples %d - Accuracy = %f', num_samples, accuracy/num_samples))
                else
                    logging:info(string.format('Step %d - training perplexity = %f', model.global_step, math.exp(loss/num_nonzeros)))
                    logging:info('Saving model')
                    local model_path = paths.concat(model_dir, string.format('model-%d', model.global_step))
                    local final_model_path_tmp = paths.concat(model_dir, '.final-model.tmp')
                    local final_model_path = paths.concat(model_dir, 'final-model')
                    if model.global_step % 1000 ~= 0 then
                        model_path = final_model_path
                    end
                    model:save(model_path)
                    logging:info(string.format('Model saved to %s', model_path))
                    os.execute(string.format('cp %s %s', model_path, final_model_path_tmp))
                    os.execute(string.format('mv %s %s', final_model_path_tmp, final_model_path))

                    -- Evaluate on val data
                    logging:info(string.format('Evaluating model on %s batches of validation data', num_batches_val))
                    local val_loss = 0
                    local val_num_samples = 0
                    local val_num_nonzeros = 0
                    local val_accuracy = 0
                    local b = 1
                    while b <= num_batches_val do
                        val_batch = val_data:nextBatch(batch_size)
                        if val_batch == nil then
                            val_data:shuffle()
                        else
                            local real_batch_size = train_batch[1]:size()[1]
                            b = b+1
                            local step_loss, stats = model:step(val_batch, true, beam_size)
                            val_loss = val_loss + step_loss
                            val_num_samples = val_num_samples + real_batch_size
                            val_num_nonzeros = val_num_nonzeros + stats[1]
                            val_accuracy = val_accuracy + stats[2]
                        end
                    end
                    logging:info(string.format('Step %d - Val Accuracy = %f', model.global_step, val_accuracy/val_num_samples))
                    num_seen = 0
                    num_nonzeros = 0
                    loss = 0
                    accuracy = 0
                    collectgarbage()
                end
            end
        end -- while true
        if forward_only then
            logging:info(string.format('Epoch: %d Number of samples %d - Accuracy = %f', epoch, num_samples, accuracy/num_samples))
        end
    end -- for epoch
end

function main()
    -- Parse command line 
    opt = cmd:parse(arg)

    logging = logger(opt.log_path)

    local phase= opt.phase
    local batch_size = opt.batch_size
    local num_epochs = opt.num_epochs

    local model_dir = opt.model_dir
    local load_model = opt.load_model
    local steps_per_checkpoint = opt.steps_per_checkpoint
    local num_batches_val = opt.num_batches_val
    local beam_size = opt.beam_size

    local gpu_id = opt.gpu_id
    local seed = opt.seed

    if gpu_id > 0 then
        logging:info(string.format('Using CUDA on GPU %d', gpu_id))
        require 'cutorch'
        require 'cunn'
        cutorch.setDevice(gpu_id)
        cutorch.manualSeed(seed)
        use_cuda = true
    else
        logging:info(string.format('Using CPU'))
        use_cuda = false
    end

    -- Build model
    logging:info('Building model')
    local model = Model()
    local final_model = paths.concat(model_dir, 'final-model')
    if load_model and paths.filep(final_model) then
        logging:info(string.format('Loading model from %s', final_model))
        model:load(final_model, opt)
    else
        logging:info('Creating model with fresh parameters')
        model:create(opt)
    end
    if not paths.dirp(model_dir) then
        paths.mkdir(model_dir)
    end

    -- Load data
    logging:info(string.format('Data base dir %s', opt.data_base_dir))
    logging:info(string.format('Load training data from %s', opt.data_path))
    local train_data = DataGen(opt.data_base_dir, opt.data_path, 10.0)
    logging:info(string.format('Training data loaded from %s', opt.data_path))
    local val_data
    if phase == 'train' then
        logging:info(string.format('Load validation data from %s', opt.val_data_path))
        val_data = DataGen(opt.data_base_dir, opt.val_data_path, 10.0)
        logging:info(string.format('Validation data loaded from %s', opt.val_data_path))
    end

    train(model, phase, batch_size, num_epochs, train_data, val_data, model_dir, steps_per_checkpoint, num_batches_val, beam_size)

    logging:shutdown()
end

main()
