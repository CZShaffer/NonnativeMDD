import matplotlib.pyplot as plt
import pathlib
import numpy as np

MODELS = ("wav2vec", "hubert")
SEEDS = [str(file.stem) for file in pathlib.Path(MODELS[0]).iterdir()]


def main():
    training_behaviors = get_training_behaviors()
    plot_losses(training_behaviors)
    write_summary(training_behaviors)


def get_training_behaviors():
    train_logs = {}
    for model in MODELS:
        results_directory = pathlib.Path(model)
        model_train_logs = []
        for seed in SEEDS:
            with open(results_directory / seed / "train_log.txt") as file:
                model_train_logs.append(file.readlines())
        train_logs[model] = model_train_logs
    training_behaviors = {}
    for model_name, runs in train_logs.items():
        train_losseses = []
        validation_losseses = []
        validation_terses = []
        test_losses = []
        test_ters = []
        for run in runs:
            train_losses = []
            validation_losses = []
            validation_ters = []
            for line in run:
                split = line.split()
                if "test" in split:
                    test_losses.append(float(split[6].strip(",")))
                    test_ters.append(float(split[9]))
                if line.startswith("e"):
                    train_losses.append(float(split[9].strip(",")))
                    validation_losses.append(float(split[13].strip(",")))
                    validation_ters.append(float(split[22].strip(",")))
            train_losseses.append(train_losses)
            validation_losseses.append(validation_losses)
            validation_terses.append(validation_ters)
        training_behaviors[model_name] = {
            "train loss": train_losseses,
            "validation loss": validation_losseses,
            "validation ter": validation_terses,
            "test losses": test_losses,
            "test ters": test_ters,
        }
    return training_behaviors


def plot_losses(training_behaviors):
    plt.title("Average Train and Validation Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    epochs = [str(x) for x in range(1, len(SEEDS) + 1)]
    losses = np.array(training_behaviors[MODELS[0]]["train loss"])
    for i in range(len(losses)):
        plt.plot(epochs, losses[i], color="tab:blue", alpha=0.1)
    plt.plot(epochs, losses.mean(0), color="tab:blue", label="wav2vec 2.0 train")
    losses = np.array(training_behaviors[MODELS[0]]["validation loss"])
    for i in range(len(losses)):
        plt.plot(epochs, losses[i], color="tab:green", alpha=0.1)
    plt.plot(epochs, losses.mean(0), color="tab:green", label="wav2vec 2.0 validation")
    losses = np.array(training_behaviors[MODELS[1]]["train loss"])
    for i in range(len(losses)):
        plt.plot(epochs, losses[i], color="tab:orange", alpha=0.1)
    plt.plot(epochs, losses.mean(0), color="tab:orange", label="HuBERT train")
    losses = np.array(training_behaviors[MODELS[1]]["validation loss"])
    for i in range(len(losses)):
        plt.plot(epochs, losses[i], color="tab:red", alpha=0.1)
    plt.plot(epochs, losses.mean(0), color="tab:red", label="HuBERT validation")
    plt.legend()
    plt.savefig("wav2vec2_hubert_losses.pdf")
    plt.clf()


def write_summary(training_behaviors):
    with open("summary.txt", "w") as file:
        losses = np.array(training_behaviors[MODELS[0]]["test losses"])
        ters = np.array(training_behaviors[MODELS[0]]["test ters"])
        file.write(
            f"wav2vec2\n"
            f"\ttest losses: {losses}\n"
            f"\t\taverage: {losses.mean()}, standard deviation: {losses.std()}\n"
            f"\ttest ters: {ters}\n"
            f"\t\taverage: {ters.mean()}, standard deviation: {ters.std()}\n"
        )

        losses = np.array(training_behaviors[MODELS[1]]["test losses"])
        ters = np.array(training_behaviors[MODELS[1]]["test ters"])
        file.write(
            f"hubert\n"
            f"\ttest losses: {losses}\n"
            f"\t\taverage: {losses.mean()}, standard deviation: {losses.std()}\n"
            f"\ttest ters: {ters}\n"
            f"\t\taverage: {ters.mean()}, standard deviation: {ters.std()}\n"
        )


if __name__ == "__main__":
    main()
