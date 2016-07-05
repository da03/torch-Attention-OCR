require 'nn'

function createCriterion(output_size)
    local weights = torch.ones(output_size)
    weights[1] = 0
    criterion = nn.ClassNLLCriterion(weights)
    criterion.sizeAverage = false
    return criterion
end
