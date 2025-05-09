from mate_ia_one.modelNNUEpythorch import train_model_with_kfold_cv, collect_data_from_csv
import matplotlib.pyplot as plt
from itertools import product


def stat_comparation():
    sizeCSV = [333,1000,2000,5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,250000,500000]
    hidden_sizes = [[769,384,192,96]]
    dropouts = [0.1]
    input_sizes = [769]
    epochs = [100]
    random_seeds = [42]
    k_folds = [5]

    # Dizionario dei parametri
    param_lists = {
        "sizeCSV": sizeCSV,
        "hidden_sizes": hidden_sizes,
        "dropouts": dropouts,
        "input_sizes": input_sizes,
        "epochs": epochs,
        "random_seeds": random_seeds,
        "k_folds": k_folds
    }

    # Individua i parametri che hanno più di un valore
    variable_params = {name: values for name, values in param_lists.items() if len(values) > 1}

    if len(variable_params) != 1:
        print("Si attende esattamente un parametro variabile per generare il grafico.")
        return

    # Recupera il parametro variabile e i suoi valori
    varying_param, varying_values = list(variable_params.items())[0]

    results = []



    for size in sizeCSV:
        collect_data_from_csv(filename="./mate_ia_one/dataset/shuffledChessData.csv",
                              datapickename="./mate_ia_one/datapickle/chess_data_" + str(size*3) + ".pickle",
                              max_rows=size)
        collect_data_from_csv(filename="./mate_ia_one/dataset/shuffledRandomEvals.csv",
                              datapickename="./mate_ia_one/datapickle/chess_data_" + str(size*3) + ".pickle",
                              max_rows=size)
        collect_data_from_csv(filename="./mate_ia_one/dataset/shuffledTaticEvals.csv",
                              datapickename="./mate_ia_one/datapickle/chess_data_" + str(size*3)+ ".pickle",
                              max_rows=size)
        # Combina tutte le combinazioni usando product (ci sono comunque dei parametri fissi)
        for epoch, input_size, hidden_size, dropout, seed, k in product(
                epochs, input_sizes, hidden_sizes, dropouts, random_seeds, k_folds
        ):
            print(f"\nTraining con dropout={dropout}, seed={seed}, k_folds={k}")
            result = train_model_with_kfold_cv(
                datapicke=f"./mate_ia_one/datapickle/chess_data_{size*3}.pickle",
                filename=f"./weight/weight12/{size*3}_{epoch}_{input_size}.pth",
                reportname=f"./report/training/gen12/training_report_cv_{size*3}_{epoch}_{dropout}_seed{seed}.txt",
                input_size=input_size,
                hidden_sizes=hidden_size,
                dropout_rate=dropout,
                epoch=epoch,
                random_seed=seed,
                k_folds=k
            )
            results.append(result)

    # Estrai i valori di R² medio dai risultati
    avg_r2_values = [r['avg_metrics']['r2'] for r in results]

    # Poiché x_values potrebbe avere ripetizioni, estrai solo i valori corrispondenti all'iterazione "principale"
    # Ad esempio, se il parametro variabile è dropout, prendi i primi len(dropouts) valori
    x_values = varying_values

    # Costruisci il grafico utilizzando gli indici
    positions = list(range(len(x_values)))
    plt.figure(figsize=(10, 6))
    plt.bar(positions, avg_r2_values[:len(x_values)], color='skyblue')
    plt.xlabel(varying_param.capitalize())
    plt.ylabel("Average Test R²")
    plt.title(f"R² Medio sul Test Set al variare di {varying_param.capitalize()}")
    plt.xticks(positions, [str(val) for val in x_values], rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()

    # Salva il grafico includendo il nome del parametro variabile nel filename
    filename = f"./report/graph/grafico_{varying_param}.png"
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    stat_comparation()
