require 'torch'
require 'nn'
require 'paths'
require 'nnutils'
require 'optim'

--DOWNLOAD MNIST
mnist = nnutils.MNISTLoader()
inputs, labels = mnist:getTrainset()

mean = { }
stdv = { }
mean[1] = inputs[{ {}, {1}, {}, {}  }]:mean() 
print('Channel ' .. 1 .. ', Mean: ' .. mean[1])
inputs[{ {}, {1}, {}, {}  }]:add(-mean[1]) -- mean subtraction
            
stdv[1] = inputs[{ {}, {1}, {}, {}  }]:std() 
print('Channel ' .. 1 .. ', Standard Deviation: ' .. stdv[1])
inputs[{ {}, {1}, {}, {}  }]:div(stdv[1]) -- std scaling

net = nn.Sequential()
net:add(nn.SpatialConvolution(1, 6, 5, 5)) -- 1 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                        
net:add(nn.SpatialMaxPooling(2,2,2,2))     
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU())                        
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor to 1D (16x5x5 --> 16*5*5)
net:add(nn.Linear(16*5*5, 120))             -- First layer of network
net:add(nn.ReLU())                       
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())                      
net:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network 
net:add(nn.LogSoftMax())       -- converts the output to a log-probability.


local criterion = nn.ClassNLLCriterion()
local params, gradParams = net:getParameters()
local optimState = {learningRate=0.01}
local batchSize = 128

for epoch = 1, 10 do
    mnist:startEpoch(batchSize)
    for x = 1, mnist:nBatches() do
        inputsX, labelsX = mnist:getMiniBatch(x)
        local function feval(params)
            gradParams:zero()      
            local outputs = net:forward(inputsX)
            local loss = criterion:forward(outputs, labelsX)
            local dloss_doutput = criterion:backward(outputs, labelsX)
            net:backward(inputsX, dloss_doutput)
            return loss,gradParams
        end
        optim.sgd(feval, params, optimState)
    end
end

test_inputs, test_labels = mnist:getTestset()
correct = 0
for i=1, mnist:getTestsetSize() do
    local groundtruth = test_labels[i]
    local prediction = net:forward(test_inputs[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print('The accuracy when compared with MNIST test data is ' .. 100*correct/10000 .. ' % ')

