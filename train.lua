require 'torch'
require 'nn'
require 'optim'
require 'paths'

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
local si = opt.save_intervals

local color_space = 'rgb'

if channels == 1 then
	color_space = 'gray'
end

opt.seed = torch.random(1, 10000)
print('random seed set')
torch.manualSeed(opt.seed)
torch.setnumthreads(num_threads)
torch.setdefaulttensortype('torch.FloatTensor')

dataset = require 'dataset'

dataset.set_dir(opt.source_dir)
dataset.set_ext(opt.file_extensions)
dataset.set_color_space(color_space)
dataset.set_batch_size(opt.batch_size)
dataset.set_img_size(opt.img_size)
dataset.load_paths()
imglist = dataset.load_img()

function string:split(sep)
        local sep, fields = sep or ":", {}
        local pattern = string.format("([^%s]+)", sep)
        self:gsub(pattern, function(c) fields[#fields+1] = c end)
        return fields
end

local max_epoch = -1
for file in paths.files(opt.net_dir) do
    local f = file:split('_')
    if tonumber(f[1]) ~= nil and tonumber(f[1]) > max_epoch and f[2] == 'gen.t7' then
        max_epoch = tonumber(f[1])
    end
end

generator, discriminator = nil, nil

if max_epoch > 0 then
	generator = torch.load(opt.net_dir..'/'..max_epoch..'_gen.t7')
	discriminator = torch.load(opt.net_dir..'/'..max_epoch..'_dis.t7')
	print('checkpoints loaded, resume training at '..max_epoch..' epoch.')
else
	model = require 'model'
	generator = model.generator
	discriminator = model.discriminator
	max_epoch = 1
end
local BCE = nn.BCECriterion()

optimizer_gen = {
	learningRate = opt.lr,
	beta1 = opt.beta1
}

optimizer_dis = {
	learningRate = opt.lr,
	beta1 = opt.beta1
}

local input = torch.Tensor(batchsize, channels, imsize, imsize)
local noise = torch.Tensor(batchsize, z_dim, 1, 1)
local label = torch.Tensor(batchsize)

if opt.use_gpu > 0 then
	require 'cunn'
	cutorch.setDevice(opt.use_gpu)
	input = input:cuda()
	noise = noise:cuda()
	label = label:cuda()
	if pcall(require, 'cudnn') then
		require 'cudnn'
		cudnn.benchmark = true
		cudnn.convert(generator, cudnn)
		cudnn.convert(discriminator, cudnn)
	end
	discriminator = discriminator:cuda()
	generator = generator:cuda()
	BCE = BCE:cuda()
end

local g_loss, d_loss = nil, nil

local g_params, g_grads = generator:getParameters()
local d_params, d_grads = discriminator:getParameters()

local function gen_noise(batchsize, z_dim)
	local noise = torch.Tensor(batchsize, z_dim, 1, 1)
	if opt.noise == 'uniform' then
		noise:uniform(-1, 1)
	elseif opt.noise == 'normal' then
		noise:normal(0, 1)
	end
	return noise
end

-- local fake_label = torch.Tensor(batchsize):fill(0)
-- local real_label = torch.Tensor(batchsize):fill(1)
local real = 1
local fake = 0

local train_dis = function (x)
	d_grads:zero()
	local real_img = imglist:get_batch()
	input:copy(real_img)
	label:fill(real)

	local output = discriminator:forward(input) -- 
	-- print('done')
	local d_loss_real = BCE:forward(output, label)
	local fDo = BCE:backward(output, label)
	discriminator:backward(input, fDo)

    noise:copy(gen_noise(batchsize, z_dim))
	local z = generator:forward(noise)
	input:copy(z)
	label:fill(fake)

	local output = discriminator:forward(input)
	local d_loss_fake = BCE:forward(output, label)
	d_loss = d_loss_real + d_loss_fake
	local fDo = BCE:backward(output, label)
	discriminator:backward(input, fDo)

	return d_loss, d_grads
end

local train_gen = function(x)
	g_grads:zero()
	label:fill(real)
	local output = discriminator.output
	g_loss = BCE:forward(output, label)
	local fDo = BCE:backward(output, label)
	local fDg = discriminator:updateGradInput(input, fDo)
	noise:copy(gen_noise(batchsize, z_dim))
	generator:backward(noise, fDg)
	return g_loss, g_grads
end


print(generator)
print(discriminator)
--train
for ep = max_epoch, opt.epoch do
	for i = 1, math.min(imglist:size(), opt.ntrain), batchsize do
		print('trainning discriminator')
		optim.adam(train_dis, d_params, optimizer_dis)
		print('trainning generator')
		optim.adam(train_gen, g_params, optimizer_gen)
		if ((i-1)/batchsize) % 1 == 0 then
			print(('epoch: [%d][%4d / %4d]\t gen loss: %.4f, dis loss: %.4f'):format(ep, 
				(i-1)/batchsize, 
				math.floor(math.min(imglist:size(), opt.ntrain)/batchsize),
				g_loss, d_loss))
		end
	end
	if ep % si == 0 then
		print('save status...')
		paths.mkdir('checkpoints')
		d_params, d_grads, g_params, g_grads = nil, nil, nil, nil
		torch.save(opt.net_dir..'/'..ep..'_dis.t7', discriminator:clearState())
		torch.save(opt.net_dir..'/'..ep..'_gen.t7', generator:clearState())
		d_params, d_grads = discriminator:getParameters()
		g_params, g_grads = generator:getParameters()
	end
end


