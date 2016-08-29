local opt = {}
opt.train = {
	batch_size = 64,
	z_dim = 100,
	gen_filter = 64,
	dis_filter = 64,
	epoch = 250,
	lr = 0.0002,
	beta1 = 0.5,
	ntrain = math.huge,
	display = 1,
	use_gpu = 0,
	noise = 'normal',
	num_threads = 1,
	kernel_size = 4,
	stride = 2, 
	padding = 1,
	channels = 3,
	img_size = 64,
	source_dir = {'./images'},
	file_extensions = 'png',
	save_intervals = 1,
	net_dir = './checkpoints',
}

opt.visualize = {
	img_num = 36,
	batch_size = 64,
	noise = 'normal',
	net_dir = './checkpoints',
	img_size = 32,
	img_name = 'result.png',
	use_gpu = 0,
	z_dim = 100,
}

return opt