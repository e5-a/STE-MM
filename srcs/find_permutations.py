from analysis.metric import test, hutchinson_trace_estimator
from utils.loader import load_model, load_dm
from utils.merge import merge_models
from utils.seed import fix_randomseed
from utils.repair import reset_bn_stats
from search.mult import wm, ste
import logging
from networks import mlp, VGG, resnet20

import hydra
from omegaconf import OmegaConf
from utils.idgenerator import IDGenerator

import time

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name='config_rebasin', config_path='../configs')
def main(cfg):
    stg = cfg.rebasin
    fix_randomseed(stg.seed)
    out_dir = str(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    dir_list = list(sorted(OmegaConf.to_container(stg, resolve=True)['targets']))
    if stg.network == "resnet20":
        w = 16
    elif stg.network == 'VGG':
        w = 4
    else:
        w = 1

    nets = []

    for i, dir in enumerate(dir_list):
        net = eval(stg.network).Net(w=w)
        load_model(net, dir)
        nets.append(net)
        nets[-1].cuda()
        if i == 0:
            match stg.method.name:
                case 'ste':
                    dm = load_dm(stg.dataset, stg.method.batch_size)
                case _:
                    dm = load_dm(stg.dataset, 64)

            dm.setup()
            test_dl = dm.test_dataloader()
            train_dl = dm.train_dataloader()
            repair_dl = None

    if stg.dataset == "CIFAR10":
        input_shape = [3, 32, 32]
    else:
        input_shape = [1, 28, 28]

    match stg.method.name:
        case 'wm':
            for net in nets:
                net.cpu()
            start_time = time.time()
            permed_nets, wms = wm.match_wm(nets, input_shape, fast_wm=stg.fast_wm)
            end_time = time.time()
            for i, w in enumerate(wms):
                w.save(f'{out_dir}/wm_{i}.pth')
        case 'ste':
            for net in nets:
                net.cuda()
            start_time = time.time()
            permed_nets, wms = ste.ste(
                    nets[0], nets[1:], train_dl, test_dl,
                    stg.method.learning_rate,
                    stg.method.num_epochs,
                    stg.method.mid,
                    'cuda:0',
                    input_shape,
                    stg.fast_wm)
            end_time = time.time()
            for i, w in enumerate(wms):
                w.save(f'{out_dir}/wm_{i}.pth')
        case _:
            assert True

    print('elapsed_time:', end_time - start_time)
    merged_model = merge_models(
            permed_nets, reset_bn=True, loader=train_dl)
    for i, net in enumerate(permed_nets):
        print(f'loss and acc of {i}th model:', test(net, test_dl))
        print(f'sharpness of {i}th model', hutchinson_trace_estimator(net, test_dl, weighted=True)[1])
    print('loss and acc of merged model', test(merged_model, test_dl))
    print(f'sharpness of merged model', hutchinson_trace_estimator(merged_model, test_dl, weighted=True)[1])


if __name__ == '__main__':
    id_generator = IDGenerator()
    OmegaConf.register_new_resolver("experiment_id", id_generator.generate)
    main()
