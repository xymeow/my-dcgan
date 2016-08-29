require 'torch'
require 'nn'

local model = {}

local opt = require 'options'
opt = opt.train
local channels = opt.channels
local z_dim = opt.z_dim
local gf = opt.gen_filter
local df = opt.dis_filter
local ks = opt.kernel_size
local stride = opt.stride
local pad = opt.padding
local imsize = opt.img_size
local batchsize = opt.batch_size

local function weights_init(m)
	local name = torch.type(m)
	if name:find('Convolution')
		then
		m.weight:normal(0.0, 0.02)
		m:noBias()
	elseif name:find('BatchNormalization')
		then
		if m.weight
			then 
			m.weight:normal(1.0, 0.02)
		end
		if m.bias
			then
			m.bias:fill(0)
		end
	end
end

local BatchNormalization = nn.SpatialBatchNormalization
local Deconvolution = nn.SpatialFullConvolution
local Convolution = nn.SpatialConvolution
local LeakyReLU = nn.LeakyReLU
local ELU = nn.ELU
local PReLU = nn.PReLU
local ReLU = nn.ReLU
local tanh = nn.Tanh
local sigmoid = nn.Sigmoid

local generator = nn.Sequential()

generator:add(Deconvolution(z_dim, gf*8, ks, ks))
generator:add(BatchNormalization(gf*8)):add(ELU())
generator:add(Deconvolution(gf*8, gf*4, ks, ks, stride, stride, pad, pad))
generator:add(BatchNormalization(gf*4)):add(ELU())
generator:add(Deconvolution(gf*4, gf*1, ks, ks, stride, stride, pad, pad))
generator:add(BatchNormalization(gf)):add(ELU())
generator:add(Deconvolution(gf, channels, ks, ks, stride, stride, pad, pad))
generator:add(tanh())

generator:apply(weights_init)

local discriminator = nn.Sequential()

discriminator:add(Convolution(channels, df, ks, ks, stride, stride, pad, pad))
discriminator:add(ELU())
discriminator:add(Convolution(df, df*4, ks, ks, stride, stride, pad, pad))
discriminator:add(BatchNormalization(df*4)):add(ELU())
discriminator:add(Convolution(df*4, df*8, ks, ks, stride, stride, pad, pad))
discriminator:add(BatchNormalization(df*8)):add(ELU())
discriminator:add(Convolution(df*8, 1, ks, ks))
discriminator:add(sigmoid())
discriminator:add(nn.View(1):setNumInputDims(3))

discriminator:apply(weights_init)

model.generator = generator
model.discriminator = discriminator

return model