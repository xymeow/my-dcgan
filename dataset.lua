require 'image'
require 'paths'
require 'torch'

local dataset = {}

dataset.height = 32
dataset.width = 32

dataset.color_space = 'rgb'
dataset.dir = {}

dataset.imgpath = nil
dataset.channels = 3
dataset.ext = ''
dataset.batch_size = 64

function dataset.set_dir(dir)
	dataset.dir = dir
end

function dataset.set_color_space(cs)
	dataset.color_space = cs
	if cs == 'gray' then
		dataset.channels = 1
	end
end

function dataset.set_ext(ext)
	dataset.ext = ext
end

function dataset.set_batch_size(batch_size)
	dataset.batch_size = batch_size
end

function dataset.load_paths()

	local files = {}
	local dirs = dataset.dir
	local ext = dataset.ext

	for i = 1, #dirs do
		local dir = dirs[i]
		for file in paths.files(dir) do
			if file:find(ext .. '$') then
				table.insert( files, paths.concat(dir, file) )
			end
		end
	end
	dataset.imgpath = files
end

function dataset.load_img()
	-- body
	if dataset.imgpath == nil then
		dataset.loadPaths()
	end
	local N = #dataset.imgpath
	local imglist = torch.FloatTensor(N, dataset.channels, dataset.height, dataset.width)
	for i = 1, N do
		local img = image.load(dataset.imgpath[i], dataset.channels, 'float')
		img = image.scale(img, dataset.width, dataset.height)
		imglist[i] = img
	end
	local result = {}
	result.data = imglist
	function result:size()
		return N
	end
	setmetatable(result, {
		__index = function(self, index) return self.data[index] end,
		__len = function(self) return self.data.size(1) end
		})
	function result:get_batch()
		local randindex = torch.randperm(N)
		-- print(dataset.color_space)
		local batch = torch.FloatTensor(dataset.batch_size, dataset.channels, dataset.height, dataset.width)
		-- print(batch_size)
		for i = 1, dataset.batch_size do
			-- table.insert(batch, imglist[randindex[i]])
			batch[i] = imglist[randindex[i]]
		end
		return batch
	end
	return result
end

return dataset