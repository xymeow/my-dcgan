require 'torch'
require 'nn'
require 'paths'
require 'image'
local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')

function string:split(sep)
        local sep, fields = sep or ":", {}
        local pattern = string.format("([^%s]+)", sep)
        self:gsub(pattern, function(c) fields[#fields+1] = c end)
        return fields
end

local opt = require 'options'
opt = opt.visualize

assert(opt.net_dir ~= '', 'provide a dir')

-- _file = {}
local max_epoch = -1
for file in paths.files(opt.net_dir) do
    local f = file:split('_')
    -- print(f)
    if tonumber(f[1]) ~= nil and tonumber(f[1]) > max_epoch and f[2] == 'gen.t7' then
        max_epoch = tonumber(f[1])
    end
end

net = torch.load(opt.net_dir..'/'..max_epoch..'_gen.t7')
if torch.type(net:get(1)) == 'nn.View' then
    net:remove(1)
end
print(net)
print('trained epoches: '..max_epoch)
noise = torch.Tensor(opt.img_num, opt.z_dim, 1, 1)

if opt.noise == 'uniform' then
    noise:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise:normal(0, 1)
end
local sample_input = torch.randn(2,100,1,1)

sample_input = sample_input:float()
net:float()
optnet.optimizeMemory(net, sample_input)

local img = net:forward(noise)
img:add(1):mul(0.5)
image.save(opt.img_name, image.toDisplayTensor(img))
