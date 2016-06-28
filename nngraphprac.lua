require 'torch'
require 'nn'
require 'paths'
require 'nnutils'
require 'optim'

--DOWNLOAD MNIST
mnist = nnutils.MNISTLoader()
inputs, labels = mnist:getTrainset()
print('reached')

mean = { }
stdv = { }
mean[1] = inputs[{ {}, {1}, {}, {}  }]:mean() -- mean estimation
print('Channel ' .. 1 .. ', Mean: ' .. mean[1])
inputs[{ {}, {1}, {}, {}  }]:add(-mean[1]) -- mean subtraction
            
stdv[1] = inputs[{ {}, {1}, {}, {}  }]:std() -- std estimation
print('Channel ' .. 1 .. ', Standard Deviation: ' .. stdv[1])
inputs[{ {}, {1}, {}, {}  }]:div(stdv[1]) -- std scaling

net = nn.Sequential()
net:add(nn.SpatialConvolution(1, 6, 5, 5))
net:add(nn.ReLU())                        
net:add(nn.SpatialMaxPooling(2,2,2,2))    
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU())                       
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))                    
net:add(nn.Linear(16*5*5, 120))            
net:add(nn.ReLU())                     
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())                      
net:add(nn.Linear(84, 10))                 
net:add(nn.LogSoftMax())      


local criterion = nn.ClassNLLCriterion()
local params, gradParams = net:getParameters()
local optimState = {learningRate=0.01}

for epoch=1,1 do
    local function feval(params)
        gradParams:zero()      
        local outputs = net:forward(inputs)
        local loss = criterion:forward(outputs, labels)
        local dloss_doutput = criterion:backward(outputs, labels)
        net:backward(inputs, dloss_doutput)
        return loss,gradParams
    end
    print('-------')
    print(epoch)
    optim.sgd(feval, params, optimState)
end
print('boutta test')
correct = 0
for i=1,10000 do
    local groundtruth = labels[i]
    local prediction = net:forward(inputs[i])
    local confidences, indices = torch.sort(prediction, true)  
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/10000 .. ' % ')

