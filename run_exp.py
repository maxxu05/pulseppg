import argparse
import torch
import os
import csv

# project utilities: logging, data loading, dynamic model import, init helpers, timer, parameter counting
from utils.utils import printlog, load_data, import_model, init_dl_program, RepeatedTimer, count_parameters
from experiments.utils_downstream import eval_classification, eval_cluster

# experiment configurations grouped by module
from experiments.configs.crossmodalrebardist_expconfigs import allcrossmodalrebardist_expconfigs
from experiments.configs.relcon_expconfigs import allrelcon_expconfigs
from experiments.configs.papagei_expconfigs import allpapagei_expconfigs
from experiments.configs.papagei_moods_expconfigs import allpapageimoods_expconfigs

# merge all experiment config dicts into one lookup
all_expconfigs = {**allcrossmodalrebardist_expconfigs, **allrelcon_expconfigs, 
                  **allpapagei_expconfigs, **allpapageimoods_expconfigs}
                    
# suppress noisy sklearn / future / user warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ensure PyTorch multiprocessing uses file-system sharing (avoids some fork/sharing issues)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


from utils.GPUtil import showUtilization

# function used by RepeatedTimer to log GPU/CPU utilization periodically
def repeating_func(path):
    printlog(showUtilization(all=True), path=path)

if __name__ == "__main__":
    
    # CLI args for selecting config and controlling retrain/eval behavior
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Select specific config from experiments/configs/",
                        type=str)
    parser.add_argument("--retrain", help="WARNING: Retrain model config, overriding existing model directory",
                        action='store_true', default=False)
    parser.add_argument("--retrain_eval", help="WARNING: Retrain eval model config, overriding existing model directory",
                        action='store_true', default=False)
    parser.add_argument("--eval_epoch", help="Epoch to be reloaded for evaluation",type=str,
                        default="best")
    parser.add_argument("--gputiloff", help="print log the gpu utilizaiton",
                        action='store_true', default=False)
    parser.add_argument("--resume_on", help="resume unfinished model training",
                        action='store_true', default=False)
    args = parser.parse_args()

    # load selected config by name and set its run directory
    CONFIGFILE = args.config
    config = all_expconfigs[CONFIGFILE]
    config.set_rundir(CONFIGFILE)

    # initialize device/thread settings for deterministic DL program startup
    init_dl_program(config=config, device_name=0, max_threads=torch.get_num_threads())

    # load train/val/test datasets and preprocessing details from config
    train_data, train_labels, val_data, val_labels, test_data, test_labels, data_normalizer, data_clipping  = \
        load_data(data_config = config.data_config)

    # import/instantiate model defined in config; pass data/meta so model can build appropriate heads/loaders
    model = import_model(config, 
                        train_data=train_data, train_labels=train_labels, 
                        val_data=val_data, val_labels=val_labels, 
                        test_data=test_data, test_labels=test_labels, 
                        data_normalizer=data_normalizer, data_clipping=data_clipping, resume_on = args.resume_on)

    # count trainable parameters in the network and print
    table, total_params = count_parameters(model.net)
    print(f"Total Trainable Params: {total_params:,}")

    try:
        # prepare log path under experiments/out/<run_dir>
        logpath = os.path.join("experiments/out", config.run_dir)
        printlog(f"----------------------------------------------------------------------------------- Config: {CONFIGFILE} -----------------------------------------------------------------------------------", logpath)

        # optionally start periodic GPU utilization logging every 5s, auto-kill after 30s
        if not args.gputiloff:
            rt = RepeatedTimer(5, repeating_func, path=logpath, killafter=30)

        # If retrain requested or no checkpoint exists, train the model
        if (args.retrain == True) or (not os.path.exists(os.path.join("experiments/out/", 
                                                                config.run_dir, 
                                                                "checkpoint_best.pkl"))):
            model.fit()

        # prepare to collect eval results into a CSV-friendly row
        all_eval_results_title = ["name", "epoch", "notes"]
        all_eval_results = [CONFIGFILE, args.eval_epoch, f"{total_params:,}"]

        # loop over evaluation configurations defined in the main config
        for eval_config in config.eval_configs:
            print(f"Doing {eval_config.model_file} evaluation")

            # load data for this eval configuration (might differ from training data)
            train_data, train_labels, val_data, val_labels, test_data, test_labels, data_normalizer, data_clipping  = \
                load_data(data_config = eval_config.data_config)
            
            # set run directory for the eval config so artifacts are stored separately
            eval_config.set_rundir(os.path.join(CONFIGFILE, eval_config.name, eval_config.model_file))

            # import the evaluation model (often a wrapper that will run a trained network)
            evalmodel = import_model(eval_config, 
                                    train_data=train_data, train_labels=train_labels, 
                                    val_data=val_data, val_labels=val_labels, 
                                    test_data=test_data, test_labels=test_labels,
                                    reload_ckpt = False, evalmodel=True)
            
            # load the trained model (from the original training config) to supply weights for evaluation
            model = import_model(config)
            model.load(args.eval_epoch)
            evalmodel.setup_eval(trained_net=model.net)

            # train the evalmodel if requested or if no checkpoint present for the eval run
            if (args.retrain_eval == True) or (not os.path.exists(os.path.join(evalmodel.run_dir, "checkpoint_best.pkl"))):
                evalmodel.fit()

            # run evaluation on the test set and get a dict of metrics
            out_test = evalmodel.test()
            printlog(eval_config.name + " " + eval_config.model_file +" ++++++++++++++++++++++++++++++++++++++++", logpath)

            # append metric names and values to the CSV row lists
            all_eval_results_title.extend(list(out_test.keys()))
            all_eval_results.extend(list(out_test.values()))

        # write a CSV with titles then results (easy to paste into spreadsheets)
        csv_file = os.path.join(logpath, f"{CONFIGFILE}_easy_paste.csv")
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(all_eval_results_title)
            writer.writerow(all_eval_results)
            
    except Exception as e:
        # re-raise exception so stacktrace is visible (could log or handle differently)
        raise  
    finally:
        # always log config and stop the repeating timer if it was started
        printlog(f"Config: {CONFIGFILE}", logpath)
        if not args.gputiloff:
            rt.stop()