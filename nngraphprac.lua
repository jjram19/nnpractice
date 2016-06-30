require 'torch'
require 'nn'
require 'paths'
require 'nnutils'
require 'optim'

--DOWNLOAD MNIST
mnist = nnutils.MNISTLoader()
inputs, labels = mnist:getTrainset()

--NORM DATA
mean = { }
stdv = { }
mean[1] = inputs[{ {}, {1}, {}, {}  }]:mean() -- mean estimation
print('Channel ' .. 1 .. ', Mean: ' .. mean[1])
inputs[{ {}, {1}, {}, {}  }]:add(-mean[1]) -- mean subtraction
            
stdv[1] = inputs[{ {}, {1}, {}, {}  }]:std() -- std estimation
print('Channel ' .. 1 .. ', Standard Deviation: ' .. stdv[1])
inputs[{ {}, {1}, {}, {}  }]:div(stdv[1]) -- std scaling

--NEURAL NETWORK
net = nn.Sequential()
net:add(nn.SpatialConvolution(1, 6, 5, 5)) -- 1 input image channels, 6 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())       -- converts the output to a log-probability. Useful for classification problems

--VARIABLE DEFS
local criterion = nn.ClassNLLCriterion()
local params, gradParams = net:getParameters()
local optimState = {learningRate=0.01}

local batchSize = 128

--TRAINING LOOP
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

--TEST DATA
test_inputs, test_labels = mnist:getTestset()
local prediction = net:forward(test_inputs)
local topscores = nnutils.topkScore(prediction, test_labels, 1)
print(topscores, 'The accuracy when compared with MNIST test data is ' .. 100*topscores/10000 .. ' % ')
