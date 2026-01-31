from misc.require_lib import *


def warning_print(s):
    print('\033[5;33m%s\033[0m' % str(s))


class Params():
    def __init__(self, json_path):
        self.update(json_path)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


def read_config(config):
    params = Params(config)
    return params


def save_codes_and_config(config, model_dir):
    if os.path.isdir(os.path.join(model_dir, "nnet")):
        print("[LOG] - Save backup to %s" % os.path.join(model_dir, ".backup"))
        if os.path.isdir(os.path.join(model_dir, ".backup")):
            print("[LOG] - The dir %s exisits. Delete it and continue." % os.path.join(model_dir, ".backup"))
            shutil.rmtree(os.path.join(model_dir, ".backup"))
        os.makedirs(os.path.join(model_dir, ".backup"))
        if os.path.exists(os.path.join(model_dir, "nnet")):
            shutil.move(os.path.join(model_dir, "nnet"), os.path.join(model_dir, ".backup/"))
        if os.path.exists(os.path.join(model_dir, "checkpoint")):
            shutil.move(os.path.join(model_dir, "checkpoint"), os.path.join(model_dir, ".backup/"))
        if os.path.exists(os.path.join(model_dir, "log")):
            shutil.move(os.path.join(model_dir, "log"), os.path.join(model_dir, ".backup/"))

    os.makedirs(os.path.join(model_dir, "log"))
    os.makedirs(os.path.join(model_dir, "checkpoint"))

    if not os.path.isdir(os.path.join(model_dir, "nnet")):
        os.makedirs(os.path.join(model_dir, "nnet"))

    shutil.copyfile(config, os.path.join(model_dir, "nnet", "config.json"))

    print("[LOG] - Train the models from scratch.")
    params = Params(config)
    return params


def padding_samples(sig, n_samples):
    if sig.shape[0] < n_samples:
        padding_size = n_samples - sig.shape[0]
        repeat_time = (padding_size + sig.shape[0] - 1) // sig.shape[0]
        padding_sig = None
        for _ in range(repeat_time):
            if padding_sig is None:
                padding_sig = sig.copy()
            else:
                padding_sig = np.concatenate((padding_sig, sig), axis=0)
        sig = np.concatenate((padding_sig, sig), axis=0)
        sig = sig[-n_samples:]
    else:
        sig = sig[-n_samples:]
    return sig


def _calc_alpha(speech, noise, snr):
    alpha = np.sqrt(np.sum(speech ** 2.0) / (np.sum(noise ** 2.0) * (10.0 ** (snr / 10.0))))
    return alpha


def _add_noise(noise_dataset, raw_data, snr):
    noise_file = random.sample(noise_dataset, 1)
    noise_raw, _ = sf.read(noise_file[0])
    while len(noise_raw) < len(raw_data):
        noise_file = random.sample(noise_dataset, 1)
        noise_raw, _ = sf.read(noise_file[0])
    max_idx = len(noise_raw) - len(raw_data)
    idx = random.randint(0, max_idx)
    noise = noise_raw[idx:idx + len(raw_data)] + 1e-7
    assert len(noise) == len(raw_data)
    noisy_data = raw_data + noise * _calc_alpha(speech=raw_data, noise=noise, snr=snr)
    return noisy_data, noise


def add_noise(noise_dataset, raw_data, low_snr, high_snr):
    snr = random.randint(low_snr, high_snr)
    noisy_data, noise = _add_noise(noise_dataset=noise_dataset, raw_data=raw_data, snr=snr)
    return noisy_data, noise


def add_rir(rir_dataset, raw_data):
    rir_file_list = random.sample(rir_dataset, 1)
    rir_file = rir_file_list[0]
    rir, _ = sf.read(rir_file)
    if len(rir.shape) == 2:
        rir_ch_index = random.randint(0, rir.shape[1] - 1)
        rir = rir[:, rir_ch_index]
    rir_index = np.argmax(rir)  # or 0
    noisy_data = signal.fftconvolve(raw_data, rir, mode='full')[rir_index:rir_index + raw_data.size]
    return noisy_data


def augment_sig(noise_dataset, rir_dataset, raw_data, low_snr, high_snr):
    random_seed = random.randint(0, 100)
    if random_seed > 50 and random_seed < 100:  # 之前加噪比例太大了了，达到了70%
        noisy_data, _ = add_noise(noise_dataset=noise_dataset, raw_data=raw_data, low_snr=low_snr, high_snr=high_snr)
    elif random_seed > 30 and random_seed < 50:
        noisy_data = add_rir(rir_dataset=rir_dataset, raw_data=raw_data)
    else:
        noisy_data = raw_data
    return noisy_data


def augment_sig_se(noise_dataset, raw_data, low_snr, high_snr):
    mix, noise = add_noise(noise_dataset=noise_dataset, raw_data=raw_data, low_snr=low_snr, high_snr=high_snr)
    return mix, noise

