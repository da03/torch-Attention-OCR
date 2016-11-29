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
cmd:option('-steps_per_checkpoint', 1000, [[Checkpointing (print perplexity, save model) per how many steps]])
cmd:option('-num_batches_val', math.huge, [[Number of batches to evaluate.]])
cmd:option('-beam_size', 1, [[Beam size.]])
cmd:option('-use_dictionary', false, [[Use dictionary during decoding or not.]])
cmd:option('-allow_digit_prefix', false, [[During decoding, allow arbitary digits before word.]])
cmd:option('-dictionary_path', '/n/rush_lab/data/image_data/train_dictionary.txt', [[The path containing dictionary. Format per line: word]])

-- Optimization
cmd:text('')
cmd:text('**Optimization**')
cmd:text('')
cmd:option('-num_epochs', 1000, [[The number of whole data passes]])
cmd:option('-batch_size', 400, [[Batch size]])
cmd:option('-learning_rate', 0.1, [[Initial learning rate]])
cmd:option('-learning_rate_min', 0.001, [[Minimum learning rate]])
cmd:option('-lr_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease on the validation set or (ii) epoch has gone past the start_decay_at_limit]])

-- Network
cmd:option('-dropout', 0.0, [[Dropout probability]])
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
cmd:option('-visualize', false, [[Print results or not]])
cmd:option('-seed', 910820, [[Load model from model-dir or not]])
cmd:option('-max_decoder_l', 50, [[Maximum number of output targets]]) -- when evaluate, this is the cut-off length.
cmd:option('-max_encoder_l', 80, [[Maximum length of input feature sequence]]) --320*10/4-1
cmd:option('-prealloc', false, [[Use memory preallocation and sharing between cloned encoder/decoders]])

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

function train(model, phase, batch_size, num_epochs, train_data, val_data, model_dir, steps_per_checkpoint, num_batches_val, beam_size, visualize, output_dir, trie)
    local loss = 0
    local num_seen = 0
    local num_samples = 0
    local num_nonzeros = 0
    local accuracy = 0
    local forward_only
    if phase == 'train' then
        forward_only = false
    elseif phase == 'test' then
        if visualize then
            model:vis(output_dir)
        end
        forward_only = true
        num_epochs = 1
        model.global_step = 0
    else
        assert(false, 'phase must be either train or test')
    end
    local learning_rate = model.optim_state.learningRate or opt.learning_rate
    learning_rate = math.max(learning_rate, opt.learning_rate_min)
    model.optim_state.learningRate = learning_rate
    logging:info(string.format('Lr: %f', learning_rate))
    for epoch = 1, num_epochs do
        if not forward_only then
            train_data:shuffle()
        end
        while true do
            train_batch = train_data:nextBatch(batch_size)
            if train_batch == nil then
                break
            end
            local real_batch_size = train_batch[1]:size()[1]
            local step_loss, stats = model:step(train_batch, forward_only, beam_size, trie)
            logging:info(string.format('%f', math.exp(loss/num_nonzeros)))
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
                    model:save(model_path)
                    logging:info(string.format('Model saved to %s', model_path))
                    os.execute(string.format('cp %s %s', model_path, final_model_path_tmp))
                    os.execute(string.format('mv %s %s', final_model_path_tmp, final_model_path))

                    num_seen = 0
                    num_nonzeros = 0
                    loss = 0
                    accuracy = 0
                    collectgarbage()
                    -- Evaluate on val data
                    logging:info(string.format('Evaluating model on %s batches of validation data', num_batches_val))
                    local val_loss = 0
                    local val_num_samples = 0
                    local val_num_nonzeros = 0
                    local val_accuracy = 0
                    local b = 1
                    while b <= num_batches_val do
                        if b % 100 == 0 then
                            logging:info (string.format('%d',b))
                        end
                        val_batch = val_data:nextBatch(batch_size)
                        if val_batch == nil then
                            val_data:shuffle()
                            if num_batches_val >= math.huge then
                                break
                            end
                        else
                            local real_batch_size = val_batch[1]:size()[1]
                            b = b+1
                            local step_loss, stats = model:step(val_batch, true, beam_size, trie)
                            val_loss = val_loss + step_loss
                            val_num_samples = val_num_samples + real_batch_size
                            val_num_nonzeros = val_num_nonzeros + stats[1]
                            val_accuracy = val_accuracy + stats[2]
                        end
                    end
                    logging:info(string.format('Step %d - Val Accuracy = %f, loss = %f', model.global_step, val_accuracy/val_num_samples, math.exp(val_loss/val_num_nonzeros)))
                    local learning_rate = model.optim_state.learning_rate
                    if prev_val_loss ~= nil and val_loss > prev_val_loss and learning_rate > opt.learning_rate_min then
                        learning_rate = math.max(learning_rate*opt.lr_decay, opt.learning_rate_min)
                        model.optim_state.learningRate = learning_rate
                        logging:info(string.format('Decay lr, current Lr: %f', learning_rate))
                    end
                    prev_val_loss = val_loss
                end
            end
        end -- while true
        if forward_only then
            logging:info(string.format('Epoch: %d Number of samples %d - Accuracy = %f', epoch, num_samples, accuracy/num_samples))
        else
            local model_path = paths.concat(model_dir, string.format('model-%d', model.global_step))
            model:save(model_path)
            logging:info(string.format('Model saved to %s', model_path))
            -- Evaluate on val data
            logging:info(string.format('Evaluating model on %s batches of validation data', num_batches_val))
            local val_loss = 0
            local val_num_samples = 0
            local val_num_nonzeros = 0
            local val_accuracy = 0
            local b = 1
            while b <= num_batches_val do
                if b % 100 == 0 then
                    logging:info (string.format('%d',b))
                end
                val_batch = val_data:nextBatch(batch_size)
                if val_batch == nil then
                    val_data:shuffle()
                    if num_batches_val >= math.huge then
                        break
                    end
                else
                    local real_batch_size = val_batch[1]:size()[1]
                    b = b+1
                    local step_loss, stats = model:step(val_batch, true, beam_size, trie)
                    val_loss = val_loss + step_loss
                    val_num_samples = val_num_samples + real_batch_size
                    val_num_nonzeros = val_num_nonzeros + stats[1]
                    val_accuracy = val_accuracy + stats[2]
                end
            end
            logging:info(string.format('Epoch: %d, Step %d - Val Accuracy = %f, loss = %f', epoch, model.global_step, val_accuracy/val_num_samples, math.exp(val_loss/val_num_nonzeros)))
            local learning_rate = model.optim_state.learning_rate
            if prev_val_loss ~= nil and val_loss > prev_val_loss and learning_rate > opt.learning_rate_min then
                learning_rate = math.max(learning_rate*opt.lr_decay, opt.learning_rate_min)
                model.optim_state.learningRate = learning_rate
                logging:info(string.format('Decay lr, current Lr: %f', learning_rate))
            end
            prev_val_loss = val_loss
        end
    end -- for epoch
end

function main()
    -- Parse command line 
    opt = cmd:parse(arg)

    logging = logger(opt.log_path)
    logging:info('Command Line Arguments:')
    logging:info(table.concat(arg, ' '))
    logging:info('End Command Line Arguments')

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

    local visualize = opt.visualize
    local output_dir = opt.output_dir
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

    if visualize then
        if not paths.dirp(output_dir) then
            paths.mkdir(output_dir)
        end
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
    local trie = nil
    if opt.use_dictionary then
        logging:info(string.format('Load dictionary from %s', opt.dictionary_path))
        trie = loadDictionary(opt.dictionary_path, opt.allow_digit_prefix)
    end
    train(model, phase, batch_size, num_epochs, train_data, val_data, model_dir, steps_per_checkpoint, num_batches_val, beam_size, visualize, output_dir, trie)

    logging:shutdown()
    model:shutdown()
end

main()
