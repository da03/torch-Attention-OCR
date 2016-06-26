 --[[ Create LSTM unit, adapted from https://github.com/karpathy/char-rnn/blob/master/model/LSTM.lua
 -- and https://github.com/harvardnlp/seq2seq-attn/blob/master/models.lua
 --    ARGS:
 --        - `input_size`      : integer, number of input dimensions
 --        - `rnn_size`  : integer, number of hidden nodes
 --        - `n`  : integer, number of layers
 --        - `use_attention`  : boolean, use attention or not
 --        - `input_feed`  : boolean, use input feeding approach or not
 --        - `dropout`  : boolean, if true apply dropout
 --    RETURNS:
 --        - `LSTM` : constructed LSTM unit (nngraph module)
 --]]
function createLSTM(input_size, rnn_size, n, use_attention, input_feed, dropout)
  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  local offset = 0
  if use_attention > 0 then
      table.insert(inputs, nn.Identity()()) -- all context (batch_size x source_l x rnn_size)
      offset = offset + 1
      if input_feed > 1 then
          table.insert(inputs, nn.Identity()()) -- prev context_attn (batch_size x rnn_size)
          offset = offset + 1
      end
  end
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1+offset]
    local prev_c = inputs[L*2+offset]
    -- the input to this layer
    if L == 1 then x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})
    -- decode the gates
    local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(all_input_sums)
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
    local in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk)
    local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk)
    local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)
    -- decode the write inputs
    local in_transform = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(all_input_sums)
    in_transform = nn.Tanh()(in_transform)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  if use_attention > 0 then
    local top_h = outputs[#outputs]
    local decoder_out
    local decoder_attn = create_decoder_attn(rnn_size, 0)
    decoder_attn.name = 'decoder_attn'
    decoder_out = decoder_attn({top_h, inputs[2]})
    if dropout > 0 then
      decoder_out = nn.Dropout(dropout, nil, false)(decoder_out)
    end     
    table.insert(outputs, decoder_out)
  end
  return nn.gModule(inputs, outputs)
end

function create_decoder_attn(rnn_size, simple)
  -- inputs[1]: 2D tensor target_t (batch_l x rnn_size) and
  -- inputs[2]: 3D tensor for context (batch_l x source_l x input_size)
  
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  local target_t = nn.LinearNoBias(rnn_size, rnn_size)(inputs[1])
  local context = inputs[2]
  simple = simple or 0
  -- get attention
  local attn = nn.MM()({context, nn.Replicate(1,3)(target_t)}) -- batch_l x source_l x 1
  attn = nn.Sum(3)(attn)
  local softmax_attn = nn.SoftMax()
  softmax_attn.name = 'softmax_attn'
  attn = softmax_attn(attn)
  attn = nn.Replicate(1,2)(attn) -- batch_l x  1 x source_l
  
  -- apply attention to context
  local context_combined = nn.MM()({attn, context}) -- batch_l x source_l x rnn_size
  context_combined = nn.Sum(2)(context_combined) -- batch_l x rnn_size
  local context_output
  if simple == 0 then
    context_combined = nn.JoinTable(2)({context_combined, inputs[1]}) -- batch_l x rnn_size*2
    context_output = nn.Tanh()(nn.LinearNoBias(rnn_size*2, rnn_size)(context_combined))
  else
    context_output = nn.CAddTable()({context_combined,inputs[1]})
  end   
  return nn.gModule(inputs, {context_output})   
end
